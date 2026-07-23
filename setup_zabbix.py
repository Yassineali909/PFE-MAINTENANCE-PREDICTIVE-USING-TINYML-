#!/usr/bin/env python3
"""
setup_zabbix.py — PredictFlow / Vantive Tunisia

Provisionne la supervision Zabbix de l'ESP32-S3 via l'API JSON-RPC :
  - un groupe d'hotes
  - un host au nom technique attendu par le bridge (ESP32_<device>)
  - dix items de type « Zabbix trapper »
  - quatre triggers SEMANTIQUES

Compatibilite : la syntaxe des expressions de trigger a change avec Zabbix 5.4.
Le script detecte la version de l'API et emet la forme adaptee :
    < 5.4  ->  {HOST:cle.last()}<>"normal"
    >= 5.4 ->  last(/HOST/cle)<>"normal"
De meme, host.create exige une interface avant Zabbix 5.2.

Principe d'alerting (rigueur academique) : aucun trigger n'est defini sur les
amplitudes vibratoires brutes. Les defauts de roulements sont des phenomenes
frequentiels ; aucun seuil d'amplitude dans le domaine temporel n'est
physiquement significatif. Zabbix opere exclusivement sur les sorties du
modele CNN 1D embarque. Les items vibratoires sont collectes a titre
documentaire, sans declencheur.

Le script est idempotent : il peut etre relance sans creer de doublons.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=BASE_DIR / ".env")
except ImportError:
    pass

ZABBIX_URL = os.getenv("ZABBIX_URL", "http://localhost:8080/api_jsonrpc.php")
API_USER   = os.getenv("ZABBIX_API_USER", "Admin")
API_PASS   = os.getenv("ZABBIX_API_PASS", "zabbix")
DEVICE     = os.getenv("ZABBIX_DEVICE", "esp32_s3_01")

HOST_NAME    = f"ESP32_{DEVICE}"
HOST_VISIBLE = f"ESP32-S3 - {DEVICE} (PredictFlow)"
GROUP_NAME   = "PredictFlow"

# value_type : 0=float 1=char 2=log 3=unsigned 4=text
FLOAT, CHAR, LOG, UNSIGNED, TEXT = 0, 1, 2, 3, 4
TRAPPER = 2

# CHAR plutot que TEXT pour les chaines : indexe et pleinement utilisable
# dans les expressions de trigger, y compris sur les versions anciennes.
ITEMS = [
    ("tinyml.class",      "TinyML - classe predite",        CHAR,     ""),
    ("tinyml.confidence", "TinyML - score de confiance",    FLOAT,    ""),
    ("tinyml.prediction", "TinyML - indice de classe",      UNSIGNED, ""),
    ("tinyml.anomaly",    "TinyML - anomalie detectee",     UNSIGNED, ""),
    ("tinyml.latency",    "TinyML - latence d'inference",   UNSIGNED, "ms"),
    ("vibration_rms",     "Vibration - RMS de la fenetre",  FLOAT,    ""),
    ("vibration_peak",    "Vibration - valeur crete",       FLOAT,    ""),
    ("wifi_rssi",         "WiFi - RSSI",                    FLOAT,    "dBm"),
    ("bridge.heartbeat",  "Bridge - battement de coeur",    UNSIGNED, ""),
    ("ota.status",        "OTA - statut du dernier flash",  CHAR,     ""),
]


def build_triggers(modern: bool):
    """Genere les triggers dans la syntaxe correspondant a la version.

    Verifie par appel API direct sur Zabbix 4.4.6 : le moteur d'expression
    historique (< 5.4) rejette la comparaison de chaine via <> ou = sur
    .last() (ex. {HOST:key.last()}<>"normal" -> "Incorrect trigger
    expression"). La forme acceptee passe par la fonction str(x), qui vaut 1
    si la valeur contient x, 0 sinon : {HOST:key.str(x)}=0 pour "differe de",
    {HOST:key.str(x)}=1 pour "egale". Les comparaisons numeriques via last()
    ne sont pas concernees. La syntaxe moderne (>= 5.4) n'a pas cette
    restriction et garde <>/= directement.
    """

    def num_last(key):
        return f"last(/{HOST_NAME}/{key})" if modern else f"{{{HOST_NAME}:{key}.last()}}"

    def differs_from(key, value):
        if modern:
            return f'last(/{HOST_NAME}/{key})<>"{value}"'
        return f"{{{HOST_NAME}:{key}.str({value})}}=0"

    def equals(key, value):
        if modern:
            return f'last(/{HOST_NAME}/{key})="{value}"'
        return f"{{{HOST_NAME}:{key}.str({value})}}=1"

    def nodata(key, window):
        if modern:
            return f"nodata(/{HOST_NAME}/{key},{window})"
        return f"{{{HOST_NAME}:{key}.nodata({window})}}"

    return [
        {
            "description": f"[{DEVICE}] Defaut de roulement detecte par le CNN 1D",
            "expression": differs_from("tinyml.class", "normal"),
            "priority": 4,
            "comments": (
                "Le modele embarque classe la fenetre vibratoire dans une classe "
                "de defaut (ball, inner_race ou outer_race). Alerte sur la sortie "
                "semantique du modele, jamais sur un seuil d'amplitude."
            ),
        },
        {
            "description": f"[{DEVICE}] Confiance du modele insuffisante (< 0.70)",
            "expression": f'{num_last("tinyml.confidence")}<0.70',
            "priority": 2,
            "comments": (
                "Le score de confiance sous 0.70 signale une fenetre hors "
                "distribution d'entrainement. Candidate prioritaire a la "
                "validation operateur dans PredictFlow."
            ),
        },
        {
            "description": f"[{DEVICE}] Echec de mise a jour OTA",
            "expression": equals("ota.status", "FAILED"),
            "priority": 4,
            "comments": (
                "Le flash OTA a echoue : l'equipement continue d'operer sur la "
                "version precedente du modele."
            ),
        },
        {
            "description": f"[{DEVICE}] Perte du bridge d'ingestion (> 60s)",
            "expression": f'{nodata("bridge.heartbeat", "60s")}=1',
            "priority": 4,
            "comments": (
                "Aucun battement de coeur recu depuis plus de 60 secondes : le "
                "script mqtt_to_zabbix.py est arrete ou a perdu le broker. "
                "Toute la telemetrie est interrompue."
            ),
        },
    ]


class ZabbixAPI:
    """Client JSON-RPC minimal, compatible Zabbix 4.x a 7.x."""

    def __init__(self, url):
        self.url = url
        self.token = None
        self.use_bearer = False
        self.version = None
        self.major = 0
        self.minor = 0
        self._id = 0

    def call(self, method, params=None, authenticated=True):
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params if params is not None else {},
            "id": self._id,
        }

        headers = {"Content-Type": "application/json-rpc"}

        if authenticated and self.token:
            if self.use_bearer:
                headers["Authorization"] = f"Bearer {self.token}"
            else:
                payload["auth"] = self.token

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise SystemExit(f"Zabbix injoignable sur {self.url} : {e}")

        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"{method} : {err.get('message')} {err.get('data', '')}")
        return body["result"]

    def login(self, user, password):
        self.version = self.call("apiinfo.version", authenticated=False)
        parts = self.version.split(".")
        self.major, self.minor = int(parts[0]), int(parts[1])
        print(f"  API Zabbix version {self.version}")

        if (self.major, self.minor) >= (6, 4):
            self.use_bearer = True
            params = {"username": user, "password": password}
        else:
            params = {"user": user, "password": password}

        self.token = self.call("user.login", params, authenticated=False)
        return self.token

    @property
    def modern_syntax(self):
        """Syntaxe d'expression introduite avec Zabbix 5.4."""
        return (self.major, self.minor) >= (5, 4)

    @property
    def needs_interface(self):
        """host.create exige une interface avant Zabbix 5.2."""
        return (self.major, self.minor) < (5, 2)

    def logout(self):
        if self.token:
            try:
                self.call("user.logout", {})
            except Exception:
                pass


def ensure_group(api, dry_run):
    found = api.call("hostgroup.get", {"filter": {"name": [GROUP_NAME]}})
    if found:
        print(f"  Groupe '{GROUP_NAME}' deja present (id={found[0]['groupid']})")
        return found[0]["groupid"]
    if dry_run:
        print(f"  [dry-run] creerait le groupe '{GROUP_NAME}'")
        return None
    gid = api.call("hostgroup.create", {"name": GROUP_NAME})["groupids"][0]
    print(f"  Groupe '{GROUP_NAME}' cree (id={gid})")
    return gid


def ensure_host(api, groupid, dry_run):
    found = api.call("host.get", {"filter": {"host": [HOST_NAME]}})
    if found:
        print(f"  Host '{HOST_NAME}' deja present (id={found[0]['hostid']})")
        return found[0]["hostid"]
    if dry_run:
        print(f"  [dry-run] creerait le host '{HOST_NAME}'")
        return None

    params = {
        "host": HOST_NAME,
        "name": HOST_VISIBLE,
        "groups": [{"groupid": groupid}],
        "description": (
            "Noeud de detection de defauts de roulements. Alimente par "
            "mqtt_to_zabbix.py via le protocole Zabbix Sender. Inference "
            "CNN 1D int8 embarquee sur ESP32-S3 N16R8."
        ),
    }

    # Avant Zabbix 5.2, une interface est obligatoire meme si tous les items
    # sont de type trapper (aucune collecte active n'est effectuee).
    if api.needs_interface:
        params["interfaces"] = [{
            "type": 1, "main": 1, "useip": 1,
            "ip": "127.0.0.1", "dns": "", "port": "10050",
        }]
    else:
        params["interfaces"] = []

    hid = api.call("host.create", params)["hostids"][0]
    print(f"  Host '{HOST_NAME}' cree (id={hid})")
    return hid


def ensure_items(api, hostid, dry_run):
    existing = {
        i["key_"]
        for i in api.call("item.get", {"hostids": hostid, "output": ["itemid", "key_"]})
    }
    created = 0
    for key, name, vtype, units in ITEMS:
        if key in existing:
            print(f"    - {key:<18} deja present")
            continue
        if dry_run:
            print(f"    - [dry-run] creerait {key}")
            continue
        params = {
            "name": name,
            "key_": key,
            "hostid": hostid,
            "type": TRAPPER,
            "value_type": vtype,
            "history": "31d",
        }
        if units:
            params["units"] = units
        if vtype in (FLOAT, UNSIGNED):
            params["trends"] = "365d"
        api.call("item.create", params)
        print(f"    - {key:<18} cree")
        created += 1
    return created


def ensure_triggers(api, dry_run):
    triggers = build_triggers(api.modern_syntax)
    existing = {
        t["description"]
        for t in api.call("trigger.get", {
            "host": HOST_NAME,
            "output": ["description"],
        })
    }
    created = 0
    for trig in triggers:
        if trig["description"] in existing:
            print(f"    - deja present : {trig['description']}")
            continue
        if dry_run:
            print(f"    - [dry-run] {trig['expression']}")
            continue
        api.call("trigger.create", {
            "description": trig["description"],
            "expression": trig["expression"],
            "priority": trig["priority"],
            "comments": trig["comments"],
            "manual_close": 1,
        })
        print(f"    - cree : {trig['description']}")
        created += 1
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Provisionne la supervision Zabbix de l'ESP32-S3.")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche les operations sans rien modifier")
    args = parser.parse_args()

    print(f"\nCible : {ZABBIX_URL}")
    print(f"Host  : {HOST_NAME}")
    print("Mode  : simulation (aucune ecriture)\n" if args.dry_run else "")

    api = ZabbixAPI(ZABBIX_URL)

    print("[1/5] Authentification")
    try:
        api.login(API_USER, API_PASS)
        syntax = "moderne (>= 5.4)" if api.modern_syntax else "historique (< 5.4)"
        print(f"  Connecte. Syntaxe de trigger : {syntax}\n")
    except RuntimeError as e:
        print(f"  Echec : {e}")
        print("  Verifiez ZABBIX_API_USER / ZABBIX_API_PASS dans le fichier .env.")
        sys.exit(1)

    try:
        print("[2/5] Groupe d'hotes")
        groupid = ensure_group(api, args.dry_run)

        print("\n[3/5] Host")
        hostid = ensure_host(api, groupid, args.dry_run)

        print("\n[4/5] Items trappeurs")
        if hostid is None:
            for key, *_ in ITEMS:
                print(f"    - [dry-run] creerait {key}")
            n_items = 0
        else:
            n_items = ensure_items(api, hostid, args.dry_run)

        print("\n[5/5] Triggers semantiques")
        n_trig = ensure_triggers(api, args.dry_run)

        if args.dry_run:
            print("\nSimulation terminee. Relancez sans --dry-run pour appliquer.")
        else:
            print(f"\nTermine. {n_items} item(s) et {n_trig} trigger(s) crees.")
            print("Redemarrez le bridge : pm2 restart bridge")
            print("Les metriques doivent passer a processed=8, failed=0.")

    except RuntimeError as e:
        print(f"\nErreur API : {e}")
        sys.exit(1)
    finally:
        api.logout()


if __name__ == "__main__":
    main()
