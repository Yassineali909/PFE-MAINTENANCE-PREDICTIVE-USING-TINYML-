#!/usr/bin/env python3
"""
setup_zabbix_notifications.py — PredictFlow / Vantive Tunisia

Complete la chaine d'alerting mise en place par setup_zabbix.py :
  - un media type webhook (POST JSON vers un backend d'alertes externe)
  - le media correspondant sur l'utilisateur Admin (toutes severites, 24/7)
  - une action de type trigger qui notifie via ce media, avec operation de
    recuperation (message de resolution)

Compatibilite Zabbix 4.4.6 (verifie par appels API directs sur cette
instance) :
  - media type webhook : type = 4 (0=email, 1=script, 2=sms ; 4=webhook).
  - le JS embarque expose l'API historique CurlHttpRequest/.AddHeader()/
    .Post()/.Status()/Zabbix.Log() en PascalCase. Les versions >= 5.0
    renomment cette API en HttpRequest/addHeader/camelCase — ne pas copier
    un exemple de la doc recente, il echouerait silencieusement ici.
  - user.update attend le parametre "user_medias" (et non "medias", qui est
    le nom utilise en lecture par user.get / date d'apres la 5.0).
  - macro d'item : {ITEM.LASTVALUE1} (il n'existe pas de {ITEM.VALUE1} en
    4.4). {ALERT.SUBJECT}/{ALERT.MESSAGE} sont les macros reservees aux
    parametres de script d'alerte (donc valables ici), pas au texte de
    notification standard.

Reutilise le client ZabbixAPI de setup_zabbix.py ainsi que ses constantes
d'environnement (HOST_NAME, GROUP_NAME, DEVICE) pour rester coherent avec le
host/groupe qu'il provisionne.

Le script est idempotent : il peut etre relance sans creer de doublons.
"""

import argparse
import os
import sys

from setup_zabbix import (
    ZabbixAPI,
    ZABBIX_URL,
    API_USER,
    API_PASS,
    GROUP_NAME,
    HOST_NAME,
    DEVICE,
)

MEDIA_TYPE_NAME = "PredictFlow Alert Webhook"
ACTION_NAME = f"[{DEVICE}] Notification PredictFlow (webhook)"

ALERT_WEBHOOK_URL = os.getenv(
    "ALERT_WEBHOOK_URL", "http://192.168.137.100:3001/api/alerts/zabbix"
)
# Le webhook s'execute DANS le conteneur Zabbix : "localhost" y designerait
# le conteneur lui-meme, jamais l'hote Docker ni la machine du dashboard.
ALERT_WEBHOOK_TOKEN = os.getenv("ALERT_WEBHOOK_TOKEN")

WEBHOOK_SCRIPT = r"""
try {
    var params = JSON.parse(value),
        req = new CurlHttpRequest(),
        resp;

    req.AddHeader('Content-Type: application/json');
    if (params.token) {
        req.AddHeader('X-Alert-Token: ' + params.token);
    }

    var payload = JSON.stringify({
        event_id: params.event_id,
        host: params.host,
        trigger: params.trigger,
        severity: params.severity,
        status: params.status,
        subject: params.subject,
        message: params.message,
        item_value: params.item_value,
        timestamp: params.timestamp
    });

    resp = req.Post(params.url, payload);

    if (req.Status() < 200 || req.Status() >= 300) {
        throw 'HTTP ' + req.Status() + ': ' + resp;
    }

    return JSON.stringify({'tags': {'endpoint': 'predictflow'}});
}
catch (error) {
    Zabbix.Log(4, '[PredictFlow webhook] ' + error);
    throw 'PredictFlow webhook failed: ' + error;
}
""".strip()

WEBHOOK_PARAMETERS = [
    {"name": "event_id", "value": "{EVENT.ID}"},
    {"name": "host", "value": "{HOST.NAME}"},
    {"name": "trigger", "value": "{TRIGGER.NAME}"},
    {"name": "severity", "value": "{EVENT.NSEVERITY}"},
    {"name": "status", "value": "{EVENT.STATUS}"},
    {"name": "subject", "value": "{ALERT.SUBJECT}"},
    {"name": "message", "value": "{ALERT.MESSAGE}"},
    {"name": "item_value", "value": "{ITEM.LASTVALUE1}"},
    {"name": "timestamp", "value": "{EVENT.DATE} {EVENT.TIME}"},
    {"name": "url", "value": ALERT_WEBHOOK_URL},
    {"name": "token", "value": ALERT_WEBHOOK_TOKEN or ""},
]

PROBLEM_SUBJECT = "[PredictFlow] {EVENT.SEVERITY} - {TRIGGER.NAME}"
PROBLEM_MESSAGE = (
    "Hote : {HOST.NAME}\n"
    "Trigger : {TRIGGER.NAME}\n"
    "Severite : {EVENT.SEVERITY}\n"
    "Valeur de l'item : {ITEM.LASTVALUE1}\n"
    "Evenement : {EVENT.ID}\n"
    "Date : {EVENT.DATE} {EVENT.TIME}\n"
    "Statut : {EVENT.STATUS}"
)
RECOVERY_SUBJECT = "[PredictFlow] Resolu - {TRIGGER.NAME}"
RECOVERY_MESSAGE = (
    "Hote : {HOST.NAME}\n"
    "Trigger : {TRIGGER.NAME}\n"
    "Evenement : {EVENT.ID}\n"
    "Resolu.\n"
    "Statut : {EVENT.STATUS}"
)


def ensure_media_type(api, dry_run):
    """Cree le media type s'il n'existe pas, sinon le met a jour en place.

    mediatype.update (et non un delete+create) est indispensable : l'action
    et le media de l'utilisateur Admin referencent ce mediatypeid, une
    recreation romprait ces references.
    """
    found = api.call("mediatype.get", {"filter": {"name": [MEDIA_TYPE_NAME]}})

    params = {
        "name": MEDIA_TYPE_NAME,
        "type": 4,
        "status": 0,
        "timeout": "10s",
        "process_tags": 0,
        "show_event_menu": 0,
        "script": WEBHOOK_SCRIPT,
        "parameters": WEBHOOK_PARAMETERS,
        "description": (
            "Relais webhook des evenements Zabbix vers le backend d'alertes "
            "PredictFlow (POST JSON, en-tete X-Alert-Token)."
        ),
    }

    if found:
        mtid = found[0]["mediatypeid"]
        if dry_run:
            print(f"  [dry-run] mettrait a jour le media type '{MEDIA_TYPE_NAME}' "
                  f"(id={mtid}) via mediatype.update")
            return mtid
        params["mediatypeid"] = mtid
        api.call("mediatype.update", params)
        print(f"  Media type '{MEDIA_TYPE_NAME}' mis a jour (id={mtid})")
        return mtid

    if dry_run:
        print(f"  [dry-run] creerait le media type '{MEDIA_TYPE_NAME}' (type=4, webhook)")
        print(f"  [dry-run] URL cible : {ALERT_WEBHOOK_URL}")
        return None

    mtid = api.call("mediatype.create", params)["mediatypeids"][0]
    print(f"  Media type '{MEDIA_TYPE_NAME}' cree (id={mtid})")
    return mtid


def ensure_user_media(api, mediatypeid, dry_run):
    users = api.call("user.get", {
        "filter": {"alias": [API_USER]},
        "selectMedias": "extend",
    })
    if not users:
        raise SystemExit(f"Utilisateur '{API_USER}' introuvable via user.get.")
    user = users[0]
    userid = user["userid"]
    existing_medias = user.get("medias", [])

    if mediatypeid is not None and any(
        m["mediatypeid"] == str(mediatypeid) for m in existing_medias
    ):
        print(f"  Media deja present sur l'utilisateur '{API_USER}' (userid={userid})")
        return userid

    if dry_run:
        print(f"  [dry-run] ajouterait le media sur l'utilisateur '{API_USER}' "
              f"(toutes severites, 1-7,00:00-24:00)")
        return userid

    new_media = {
        "mediatypeid": mediatypeid,
        "sendto": "predictflow-webhook",
        "active": 0,   # 0 = active en Zabbix (convention API, pas 1)
        "severity": 63,  # toutes les severites (bitmask 111111)
        "period": "1-7,00:00-24:00",
    }
    user_medias = existing_medias + [new_media]

    api.call("user.update", {
        "userid": userid,
        "user_medias": user_medias,
    })
    print(f"  Media ajoute sur l'utilisateur '{API_USER}' (userid={userid})")
    return userid


def ensure_action(api, groupid, mediatypeid, userid, dry_run):
    found = api.call("action.get", {"filter": {"name": [ACTION_NAME]}})
    if found:
        print(f"  Action '{ACTION_NAME}' deja presente (id={found[0]['actionid']})")
        return found[0]["actionid"]

    if dry_run:
        print(f"  [dry-run] creerait l'action '{ACTION_NAME}' (status=0, groupe '{GROUP_NAME}')")
        return None

    opmessage_common = {
        "default_msg": 0,
        "mediatypeid": mediatypeid,
    }

    params = {
        "name": ACTION_NAME,
        "eventsource": 0,   # 0 = evenements de trigger
        "status": 0,        # 0 = activee
        "esc_period": "1h",
        "def_shortdata": PROBLEM_SUBJECT,
        "def_longdata": PROBLEM_MESSAGE,
        "filter": {
            "evaltype": 0,
            "conditions": [
                {"conditiontype": 0, "operator": 0, "value": groupid},  # 0 = groupe d'hotes
            ],
        },
        "operations": [
            {
                "operationtype": 0,  # 0 = envoyer un message
                "esc_step_from": 1,
                "esc_step_to": 1,
                "esc_period": "0",
                "opmessage_usr": [{"userid": userid}],
                "opmessage": {
                    **opmessage_common,
                    "subject": PROBLEM_SUBJECT,
                    "message": PROBLEM_MESSAGE,
                },
            }
        ],
        "recovery_operations": [
            {
                "operationtype": 0,
                "opmessage_usr": [{"userid": userid}],
                "opmessage": {
                    **opmessage_common,
                    "subject": RECOVERY_SUBJECT,
                    "message": RECOVERY_MESSAGE,
                },
            }
        ],
    }
    actionid = api.call("action.create", params)["actionids"][0]
    print(f"  Action '{ACTION_NAME}' creee (id={actionid})")
    return actionid


def main():
    parser = argparse.ArgumentParser(
        description="Complete la chaine d'alerting Zabbix (media type webhook + action).")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche les operations sans rien modifier")
    args = parser.parse_args()

    print(f"\nCible : {ZABBIX_URL}")
    print(f"Host  : {HOST_NAME}  /  Groupe : {GROUP_NAME}")
    print("Mode  : simulation (aucune ecriture)\n" if args.dry_run else "")

    if not ALERT_WEBHOOK_TOKEN:
        msg = "ALERT_WEBHOOK_TOKEN absent du .env"
        if args.dry_run:
            print(f"  /!\\ {msg} (le plan sera quand meme affiche, avec token=<MANQUANT>)\n")
        else:
            raise SystemExit(
                f"{msg}. Ajoutez ALERT_WEBHOOK_TOKEN=... dans .env avant l'execution reelle."
            )

    api = ZabbixAPI(ZABBIX_URL)

    print("[1/4] Authentification")
    try:
        api.login(API_USER, API_PASS)
        print(f"  Connecte.\n")
    except RuntimeError as e:
        print(f"  Echec : {e}")
        print("  Verifiez ZABBIX_API_USER / ZABBIX_API_PASS dans le fichier .env.")
        sys.exit(1)

    try:
        print("[2/4] Groupe d'hotes")
        group = api.call("hostgroup.get", {"filter": {"name": [GROUP_NAME]}})
        if not group:
            raise SystemExit(
                f"Groupe '{GROUP_NAME}' introuvable. Lancez d'abord setup_zabbix.py."
            )
        groupid = group[0]["groupid"]
        print(f"  Groupe '{GROUP_NAME}' (id={groupid})")

        print("\n[3/4] Media type webhook")
        mediatypeid = ensure_media_type(api, args.dry_run)

        print("\n[4/4] Media utilisateur + action")
        userid = ensure_user_media(api, mediatypeid, args.dry_run)
        ensure_action(api, groupid, mediatypeid, userid, args.dry_run)

        if args.dry_run:
            print("\nSimulation terminee. Relancez sans --dry-run pour appliquer.")
        else:
            print("\nTermine. Provoquez un trigger pour valider (trigger.get / alert.get).")

    except RuntimeError as e:
        print(f"\nErreur API : {e}")
        sys.exit(1)
    finally:
        api.logout()


if __name__ == "__main__":
    main()
