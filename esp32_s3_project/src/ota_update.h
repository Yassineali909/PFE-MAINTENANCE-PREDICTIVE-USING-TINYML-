#pragma once
/*
 * ota_update.h — Mise a jour du modele TinyML par voie radio.
 *
 * Le transfert porte sur le modele quantifie (~15 ko) et non sur l'image
 * applicative (1,1 Mo) : la boucle MLOps promeut un modele, pas un firmware.
 * Aucune etape de compilation n'intervient entre MLflow et la cible.
 *
 * Sequence : GET /firmware/version -> comparaison NVS -> GET /firmware/latest
 *            -> verification SHA-256 -> ecriture LittleFS -> publication du
 *            statut -> redemarrage.
 *
 * Securite d'exploitation : le modele embarque en PROGMEM sert de repli. Un
 * telechargement corrompu ou interrompu ne peut pas rendre la carte inoperante.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <LittleFS.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <esp_heap_caps.h>
#include "mbedtls/sha256.h"

#ifndef OTA_SERVER_URL
#define OTA_SERVER_URL "http://192.168.137.100:8090"
#endif

#define OTA_MODEL_PATH  "/model.tflite"
#define OTA_META_PATH   "/model.json"
#define OTA_TOPIC       "esp32/ota/status"
#define OTA_MAX_MODEL   (256 * 1024)

static Preferences ota_prefs;

// --- Etat du modele actif ---------------------------------------------------
struct OtaModelInfo {
    bool   from_flash_fs = false;   // true = LittleFS, false = PROGMEM
    String version       = "embedded";
    float  in_scale      = 0.0f;
    int    in_zp         = 0;
    float  out_scale     = 0.0f;
    int    out_zp        = 0;
};
static OtaModelInfo ota_info;

// --- Utilitaires ------------------------------------------------------------
static String ota_sha256_hex(const uint8_t* d, size_t n) {
    uint8_t h[32];
    mbedtls_sha256_context c;
    mbedtls_sha256_init(&c);
    mbedtls_sha256_starts(&c, 0);
    mbedtls_sha256_update(&c, d, n);
    mbedtls_sha256_finish(&c, h);
    mbedtls_sha256_free(&c);
    char buf[65];
    for (int i = 0; i < 32; i++) sprintf(buf + i * 2, "%02x", h[i]);
    buf[64] = 0;
    return String(buf);
}

static void ota_publish(PubSubClient& mqtt, const char* status,
                        const char* version, const char* detail) {
    StaticJsonDocument<320> d;
    d["device"]  = "esp32_s3_01";
    d["status"]  = status;                       // SUCCESS | FAILED | NO_UPDATE
    d["version"] = version;
    d["detail"]  = detail;
    d["ts"]      = (uint32_t)time(nullptr);
    char out[320];
    size_t n = serializeJson(d, out);
    if (mqtt.connected() && mqtt.publish(OTA_TOPIC, (const uint8_t*)out, n, false))
        Serial.printf("[OTA] Statut publie : %s (%s)\n", status, detail);
    else
        Serial.printf("[OTA] Publication du statut echouee (%s)\n", status);
}

// --- Montage du systeme de fichiers -----------------------------------------
static bool ota_fs_begin() {
    if (!LittleFS.begin(true)) {          // true : formatage au premier montage
        Serial.println("[OTA] Montage LittleFS echoue.");
        return false;
    }
    Serial.printf("[OTA] LittleFS monte — %u/%u octets utilises\n",
                  (unsigned)LittleFS.usedBytes(), (unsigned)LittleFS.totalBytes());
    ota_prefs.begin("ota", false);
    ota_info.version = ota_prefs.getString("version", "embedded");
    if (ota_info.version.length() == 0) ota_info.version = "embedded";
    Serial.printf("[OTA] Version locale lue en NVS : '%s'\n", ota_info.version.c_str());
    return true;
}

// --- Chargement du modele depuis LittleFS -----------------------------------
// Retourne un pointeur PSRAM persistant (jamais libere) ou nullptr.
static uint8_t* ota_load_model_from_fs(size_t* out_size) {
    if (!LittleFS.exists(OTA_MODEL_PATH)) {
        Serial.println("[OTA] Aucun modele en LittleFS — repli sur PROGMEM.");
        return nullptr;
    }
    File f = LittleFS.open(OTA_MODEL_PATH, "r");
    if (!f) return nullptr;
    size_t sz = f.size();
    if (sz == 0 || sz > OTA_MAX_MODEL) {
        Serial.printf("[OTA] Taille de modele invalide (%u octets).\n", (unsigned)sz);
        f.close();
        return nullptr;
    }
    uint32_t caps = psramFound() ? (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
                                 : MALLOC_CAP_8BIT;
    // Alignement 16 octets : le flatbuffer TFLite exige un buffer aligne.
    uint8_t* buf = (uint8_t*)heap_caps_aligned_alloc(16, sz, caps);
    if (!buf) {
        Serial.println("[OTA] Allocation du tampon de modele echouee.");
        f.close();
        return nullptr;
    }
    size_t rd = f.read(buf, sz);
    f.close();
    if (rd != sz) {
        Serial.println("[OTA] Lecture incomplete du modele.");
        return nullptr;
    }
    // Metadonnees : parametres de quantification transmis avec le modele.
    if (LittleFS.exists(OTA_META_PATH)) {
        File m = LittleFS.open(OTA_META_PATH, "r");
        StaticJsonDocument<512> d;
        if (deserializeJson(d, m) == DeserializationError::Ok) {
            JsonObject q = d["quantization"];
            if (!q.isNull()) {
                ota_info.in_scale  = q["input_scale"]  | 0.0f;
                ota_info.in_zp     = q["input_zp"]     | 0;
                ota_info.out_scale = q["output_scale"] | 0.0f;
                ota_info.out_zp    = q["output_zp"]    | 0;
            }
            ota_info.version = String((const char*)(d["version"] | "unknown"));
        }
        m.close();
    }
    ota_info.from_flash_fs = true;
    *out_size = sz;
    Serial.printf("[OTA] Modele charge depuis LittleFS — v%s, %u octets\n",
                  ota_info.version.c_str(), (unsigned)sz);
    Serial.printf("[OTA] Quantification : in(scale=%.8f zp=%d) out(scale=%.8f zp=%d)\n",
                  ota_info.in_scale, ota_info.in_zp,
                  ota_info.out_scale, ota_info.out_zp);
    return buf;
}

// --- Verification et telechargement -----------------------------------------
static void ota_check_and_update(PubSubClient& mqtt) {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    http.setTimeout(8000);
    if (!http.begin(String(OTA_SERVER_URL) + "/firmware/version")) return;
    int code = http.GET();
    if (code != 200) {
        if (code != 404) Serial.printf("[OTA] /firmware/version -> HTTP %d\n", code);
        http.end();
        return;                              // 404 = rien a distribuer
    }

    StaticJsonDocument<640> meta;
    DeserializationError err = deserializeJson(meta, http.getString());
    http.end();
    if (err) {
        Serial.println("[OTA] version.json illisible.");
        return;
    }

    String remote_v   = String((const char*)(meta["version"] | ""));
    String remote_sha = String((const char*)(meta["sha256"]  | ""));
    size_t remote_sz  = meta["size_bytes"] | 0;

    Serial.printf("[OTA] Comparaison : distante='%s' locale='%s' (len=%d/%d)\n",
                  remote_v.c_str(), ota_info.version.c_str(),
                  remote_v.length(), ota_info.version.length());

    if (remote_v.length() == 0 || remote_v == ota_info.version) return;  // a jour

    Serial.printf("\n[OTA] Nouvelle version disponible : %s (locale : %s)\n",
                  remote_v.c_str(), ota_info.version.c_str());

    if (remote_sz == 0 || remote_sz > OTA_MAX_MODEL) {
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "taille annoncee invalide");
        return;
    }

    // --- Telechargement en memoire ---
    if (!http.begin(String(OTA_SERVER_URL) + "/firmware/latest")) return;
    http.addHeader("X-ESP32-Version", ota_info.version);
    code = http.GET();
    if (code != 200) {
        http.end();
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "telechargement refuse");
        return;
    }

    uint32_t caps = psramFound() ? (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
                                 : MALLOC_CAP_8BIT;
    uint8_t* dl = (uint8_t*)heap_caps_malloc(remote_sz, caps);
    if (!dl) {
        http.end();
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "allocation impossible");
        return;
    }

    WiFiClient* stream = http.getStreamPtr();
    size_t got = 0;
    uint32_t t0 = millis();
    while (http.connected() && got < remote_sz && (millis() - t0) < 30000) {
        size_t avail = stream->available();
        if (avail) {
            int r = stream->readBytes(dl + got, min(avail, remote_sz - got));
            if (r > 0) { got += r; t0 = millis(); }
        } else {
            delay(5);
        }
    }
    http.end();

    if (got != remote_sz) {
        heap_caps_free(dl);
        Serial.printf("[OTA] Transfert incomplet : %u/%u octets\n",
                      (unsigned)got, (unsigned)remote_sz);
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "transfert incomplet");
        return;
    }

    // --- Verification d'integrite ---
    String local_sha = ota_sha256_hex(dl, got);
    if (!remote_sha.equalsIgnoreCase(local_sha)) {
        heap_caps_free(dl);
        Serial.printf("[OTA] SHA-256 non concordant.\n  attendu : %s\n  calcule : %s\n",
                      remote_sha.c_str(), local_sha.c_str());
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "empreinte SHA-256 non concordante");
        return;
    }
    Serial.printf("[OTA] SHA-256 verifie : %s\n", local_sha.substring(0, 16).c_str());

    // --- Ecriture ---
    File f = LittleFS.open(OTA_MODEL_PATH, "w");
    if (!f || f.write(dl, got) != got) {
        if (f) f.close();
        heap_caps_free(dl);
        ota_publish(mqtt, "FAILED", remote_v.c_str(), "ecriture LittleFS echouee");
        return;
    }
    f.close();
    heap_caps_free(dl);

    File m = LittleFS.open(OTA_META_PATH, "w");
    if (m) { serializeJson(meta, m); m.close(); }

    ota_prefs.putString("version", remote_v);
    Serial.printf("[OTA] Modele v%s installe — redemarrage.\n", remote_v.c_str());
    ota_publish(mqtt, "SUCCESS", remote_v.c_str(), "modele installe, redemarrage");
    mqtt.loop();
    delay(600);
    ESP.restart();
}
