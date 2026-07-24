// ============================================================================
//  PredictFlow — ESP32-S3 N16R8 — Detection de defauts de roulements (CWRU)
//  Modele de production : CNN 1D int8 (TFLite Micro)
//
//  Compilation conditionnelle :
//    -D ENABLE_MLP=0  -> firmware de PRODUCTION (CNN 1D seul)   [env:esp32s3_cnn]
//    -D ENABLE_MLP=1  -> firmware de BENCHMARK  (CNN 1D + MLP)  [env:esp32s3_mlp]
//
//  Le MLP n'est pas supprime : il reste compilable pour reproduire les
//  mesures on-device du tableau comparatif (latence / arene / precision).
// ============================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <esp_heap_caps.h>
#include <time.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "model_cnn_data.h"
#include "cwru_real_windows.h"
#include "ota_update.h"
#include "certs/ca_cert.h"

#ifndef ENABLE_MLP
#define ENABLE_MLP 0
#endif

#if ENABLE_MLP
  #include "model_mlp_data.h"
  #include "mlp_features.h"
#endif

// ============================================================================
//  Configuration reseau
// ============================================================================
// Identifiants WiFi et MQTT : definis dans secrets.h, exclu du depot.
// Copier secrets.h.example en secrets.h et le completer avant compilation.
#include "secrets.h"

const char* MQTT_BROKER   = "192.168.137.100";
const int   MQTT_PORT     = 8883;
const char* MQTT_TOPIC    = "factory/line1/esp32_s3_01/telemetry";
const char* DEVICE_ID     = "esp32_s3_01";

// ============================================================================
//  Classes — ordre LabelEncoder sklearn (ALPHABETIQUE, non modifiable)
//  Toute permutation ici invalide silencieusement toutes les predictions.
// ============================================================================
const int   NUM_CLASSES   = 4;
const char* CLASS_NAMES[] = {"ball", "inner_race", "normal", "outer_race"};
const int   IDX_NORMAL    = 2;

// ============================================================================
//  Simulation — cadence de changement de classe
//  20 inferences x 5 s de delay() = 100 s par classe. Un defaut de roulement
//  reel persiste sur des minutes/heures, pas 15 s (3 inferences) : ce rythme
//  rendait les paires PROBLEM/RESOLVED Zabbix peu credibles pour une
//  demonstration industrielle.
// ============================================================================
constexpr int INFERENCES_PER_CLASS = 20;

// ============================================================================
//  Parametres de quantification
//  A VALIDER contre tf.lite.Interpreter avant toute campagne de mesure :
//    it.get_input_details()[0]["quantization"]
//  Le log [QUANT-PARAMS] affiche a l'execution "lu" vs "impose" : si les deux
//  coincident, ces constantes peuvent etre remplacees par input->params.
// ============================================================================
float             CNN_IN_SCALE  = 0.048159f;   // repli du modele PROGMEM ;
                                               // ecrase si un modele OTA est charge
constexpr int32_t CNN_IN_ZP     = -2;
constexpr float   OUT_SCALE     = 0.00390625f; // 1/256 — sortie softmax int8
constexpr int32_t OUT_ZP        = -128;

#if ENABLE_MLP
constexpr float   MLP_IN_SCALE  = 0.039822947f;
constexpr int32_t MLP_IN_ZP     = -56;
#endif

// ============================================================================
//  TFLite Micro — arenes et interpreteurs
// ============================================================================
constexpr int kArenaSizeCNN = 150 * 1024;
uint8_t*                  tensor_arena_cnn = nullptr;
tflite::AllOpsResolver    resolver_cnn;
tflite::MicroInterpreter* interp_cnn = nullptr;
const tflite::Model*      model_cnn  = nullptr;

#if ENABLE_MLP
// Un resolver DEDIE par interpreteur. Partager un AllOpsResolver entre CNN et
// MLP corrompt l'etat interne des kernels FULLY_CONNECTED : le MLP renvoie
// alors une constante independante de l'entree.
constexpr int kArenaSizeMLP = 40 * 1024;
uint8_t*                  tensor_arena_mlp = nullptr;
tflite::AllOpsResolver    resolver_mlp;
tflite::MicroInterpreter* interp_mlp = nullptr;
const tflite::Model*      model_mlp  = nullptr;
#endif

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

WiFiClientSecure wifi_secure;
PubSubClient     mqtt_client(wifi_secure);
static uint32_t  msg_id = 0;
static float     signal_buffer[WINDOW_SIZE];

struct BenchmarkResult {
    int      predicted_class;
    float    confidence;
    uint32_t latency_ms;
    uint32_t arena_used;
    bool     success;
    float    scores[NUM_CLASSES];   // score dequantifie par classe, ordre CLASS_NAMES
};

// ============================================================================
//  Verification materielle N16R8 — preuve directe pour le rapport
// ============================================================================
void print_hardware_report() {
    Serial.println("\n=== Verification materielle ESP32-S3 N16R8 ===");
    Serial.printf("Puce          : %s rev%d, %d coeur(s) @ %d MHz\n",
                  ESP.getChipModel(), ESP.getChipRevision(),
                  ESP.getChipCores(), getCpuFrequencyMhz());
    Serial.printf("Flash         : %u octets (attendu 16777216)\n",
                  ESP.getFlashChipSize());
    Serial.printf("SRAM interne  : %u octets libres\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.printf("PSRAM totale  : %u octets (attendu 8388608)\n",
                  ESP.getPsramSize());
    Serial.printf("PSRAM libre   : %u octets\n", ESP.getFreePsram());
    Serial.printf("psramFound()  : %s\n", psramFound() ? "OUI" : "NON");
    Serial.println("==============================================\n");
}

// ============================================================================
//  Initialisation TFLM
// ============================================================================
bool init_tflm() {
    Serial.println("[TFLM] Verification de la PSRAM...");

    uint32_t caps = psramFound() ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    if (!psramFound()) {
        Serial.println("[ATTENTION] PSRAM non detectee — bascule en SRAM interne.");
        Serial.println("[ATTENTION] Le handshake TLS risque d'echouer (heap insuffisant).");
    } else {
        Serial.println("[OK] PSRAM detectee — allocation des arenes en SPIRAM.");
    }

    tensor_arena_cnn = (uint8_t*)heap_caps_malloc(kArenaSizeCNN, caps);
    if (!tensor_arena_cnn) {
        Serial.println("[FATAL] Allocation de l'arene CNN echouee.");
        return false;
    }

    // Priorite au modele distribue par voie radio ; le modele compile en
    // PROGMEM sert de repli si LittleFS est vide ou le fichier illisible.
    size_t   ota_sz  = 0;
    uint8_t* ota_buf = ota_load_model_from_fs(&ota_sz);

    if (ota_buf) {
        model_cnn = tflite::GetModel(ota_buf);
        if (model_cnn->version() != TFLITE_SCHEMA_VERSION) {
            Serial.println("[OTA] Schema du modele telecharge invalide - repli PROGMEM.");
            model_cnn = tflite::GetModel(cnn1d_bearing_int8_tflite);
            ota_info.from_flash_fs = false;
            ota_info.version = "embedded";
        } else if (ota_info.in_scale > 0.0f) {
            CNN_IN_SCALE = ota_info.in_scale;
            Serial.printf("[OTA] CNN_IN_SCALE reglee sur %.8f (modele v%s)\n",
                          CNN_IN_SCALE, ota_info.version.c_str());
        }
    } else {
        model_cnn = tflite::GetModel(cnn1d_bearing_int8_tflite);
    }
    Serial.printf("[TFLM] Source du modele : %s (v%s)\n",
                  ota_info.from_flash_fs ? "LittleFS (OTA)" : "PROGMEM (compile)",
                  ota_info.version.c_str());
    if (model_cnn->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[FATAL] Version de schema CNN invalide.");
        return false;
    }

    interp_cnn = new tflite::MicroInterpreter(
        model_cnn, resolver_cnn, tensor_arena_cnn, kArenaSizeCNN, error_reporter);

    if (interp_cnn->AllocateTensors() != kTfLiteOk) {
        Serial.println("[FATAL] AllocateTensors CNN echoue.");
        return false;
    }

    TfLiteTensor* in_cnn = interp_cnn->input(0);
    Serial.printf("[CNN1D] AllocateTensors OK — arena_used=%d / alloue=%d octets\n",
                  (int)interp_cnn->arena_used_bytes(), kArenaSizeCNN);
    Serial.printf("[CNN1D] input dims: [%d, %d] (attendu [1, %d])\n",
                  in_cnn->dims->data[0], in_cnn->dims->data[1], WINDOW_SIZE);

#if ENABLE_MLP
    tensor_arena_mlp = (uint8_t*)heap_caps_malloc(kArenaSizeMLP, caps);
    if (!tensor_arena_mlp) {
        Serial.println("[FATAL] Allocation de l'arene MLP echouee.");
        return false;
    }

    model_mlp = tflite::GetModel(mlp_bearing_int8_tflite);
    if (model_mlp->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[FATAL] Version de schema MLP invalide.");
        return false;
    }

    interp_mlp = new tflite::MicroInterpreter(
        model_mlp, resolver_mlp, tensor_arena_mlp, kArenaSizeMLP, error_reporter);

    if (interp_mlp->AllocateTensors() != kTfLiteOk) {
        Serial.println("[FATAL] AllocateTensors MLP echoue.");
        return false;
    }

    TfLiteTensor* in_mlp = interp_mlp->input(0);
    Serial.printf("[MLP] AllocateTensors OK — arena_used=%d / alloue=%d octets\n",
                  (int)interp_mlp->arena_used_bytes(), kArenaSizeMLP);
    Serial.printf("[MLP] input dims: [%d, %d] (attendu [1, 12])\n",
                  in_mlp->dims->data[0], in_mlp->dims->data[1]);
#endif

    Serial.println("[TFLM] Initialisation terminee avec succes.");
    return true;
}

// ============================================================================
//  Inference
// ============================================================================
BenchmarkResult run_inference(tflite::MicroInterpreter* interp,
                              const char* name, float* buf, bool is_mlp) {
    BenchmarkResult res = {0, 0.0f, 0, 0, false};
    TfLiteTensor* input  = interp->input(0);
    TfLiteTensor* output = interp->output(0);

#if ENABLE_MLP
    float   scale = is_mlp ? MLP_IN_SCALE : CNN_IN_SCALE;
    int32_t zp    = is_mlp ? MLP_IN_ZP    : CNN_IN_ZP;
#else
    (void)is_mlp;
    float   scale = CNN_IN_SCALE;
    int32_t zp    = CNN_IN_ZP;
#endif

    Serial.printf("[QUANT-PARAMS] %s: lu(scale=%.6f zp=%ld) -> impose(scale=%.6f zp=%ld)\n",
                  name, input->params.scale, (long)input->params.zero_point,
                  scale, (long)zp);

#if ENABLE_MLP
    if (is_mlp) {
        float raw[12], f[12];
        extract_mlp_features(buf, WINDOW_SIZE, raw);

        Serial.print("[FEAT] rms=");  Serial.print(raw[0], 4);
        Serial.print(" peak=");       Serial.print(raw[1], 4);
        Serial.print(" crest=");      Serial.print(raw[2], 3);
        Serial.print(" kurt=");       Serial.print(raw[6], 3);
        Serial.println();

        for (int i = 0; i < 12; i++) {
            f[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
            int32_t q = lroundf(f[i] / scale) + zp;
            input->data.int8[i] = (int8_t)constrain(q, -128, 127);
        }
    } else
#endif
    {
        // Normalisation PAR FENETRE, identique a normalize_window() du training :
        //   wn = (w - mean(w)) / (std(w) + 1e-6)
        // Sans elle le CNN recoit une distribution inconnue et predit faux.
        double sum = 0, sum2 = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            sum  += buf[i];
            sum2 += (double)buf[i] * buf[i];
        }
        float m  = (float)(sum / WINDOW_SIZE);
        float sd = sqrtf((float)(sum2 / WINDOW_SIZE - (double)m * m)) + 1e-6f;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            float wn  = (buf[i] - m) / sd;
            int32_t q = lroundf(wn / scale) + zp;
            input->data.int8[i] = (int8_t)constrain(q, -128, 127);
        }
    }

    uint32_t t0 = millis();
    if (interp->Invoke() != kTfLiteOk) {
        Serial.printf("[%s] Invoke echoue\n", name);
        return res;
    }
    res.latency_ms = millis() - t0;
    res.arena_used = interp->arena_used_bytes();

    Serial.print("[OUT-RAW] ");
    for (int i = 0; i < NUM_CLASSES; i++) Serial.printf("%d ", output->data.int8[i]);
    Serial.printf("| out_scale=%.6f out_zp=%ld\n", OUT_SCALE, (long)OUT_ZP);

    float max_score = -1e9f;
    int   max_idx   = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float score = (output->data.int8[i] - OUT_ZP) * OUT_SCALE;
        res.scores[i] = score;
        if (score > max_score) { max_score = score; max_idx = i; }
    }

    res.predicted_class = max_idx;
    res.confidence      = max_score;
    res.success         = true;

    Serial.printf("[%s] Classe: %-12s | Confiance: %.3f | Latence: %lu ms | Arene: %lu o\n",
                  name, CLASS_NAMES[max_idx], max_score,
                  (unsigned long)res.latency_ms, (unsigned long)res.arena_used);
    return res;
}

// ============================================================================
//  Publication MQTTS
//  CWRU est un signal MONO-AXE (accelerometre drive-end, cle DE_time).
//  On publie donc des statistiques de fenetre (RMS / crete / ecart-type) et
//  non trois axes vibration_x/y/z qui laisseraient croire a un capteur 3D.
// ============================================================================
void publish_telemetry(BenchmarkResult& cnn, float* buf, int sim_class
#if ENABLE_MLP
                     , BenchmarkResult& mlp
#endif
) {
    if (!mqtt_client.connected()) return;

    // Capacite du pool ArduinoJson (distincte de la taille du texte serialise,
    // verifiee separement plus bas) : mesuree hors-cible avec le "probs"
    // ajoute, le pool a 768 octets deborde deja a 768/768 (CNN seul) et
    // 768/768 en overflow avec le bloc "benchmark" du MLP. 1024 laisse une
    // marge confortable (832/1024 CNN seul, 992/1024 avec MLP).
    StaticJsonDocument<1024> doc;

    double sum = 0, sum2 = 0;
    float  peak = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        sum  += buf[i];
        sum2 += (double)buf[i] * buf[i];
        if (fabsf(buf[i]) > peak) peak = fabsf(buf[i]);
    }
    float mean = (float)(sum / WINDOW_SIZE);
    float rms  = sqrtf((float)(sum2 / WINDOW_SIZE));
    float sd   = sqrtf((float)(sum2 / WINDOW_SIZE - (double)mean * mean));

    doc["device"] = DEVICE_ID;

    // Horodatage epoch via NTP ; repli sur l'uptime si la synchro a echoue.
    time_t now = time(nullptr);
    doc["timestamp"] = (now > 1700000000)
                     ? (uint32_t)now
                     : (uint32_t)(millis() / 1000);
    doc["msg_id"] = ++msg_id;

    JsonObject s = doc.createNestedObject("sensors");
    s["vibration_rms"]  = roundf(rms  * 10000) / 10000.0f;
    s["vibration_peak"] = roundf(peak * 10000) / 10000.0f;
    s["vibration_std"]  = roundf(sd   * 10000) / 10000.0f;
    s["wifi_rssi"]      = WiFi.RSSI();

    JsonObject t = doc.createNestedObject("tinyml");
    t["prediction"]  = cnn.predicted_class;
    t["confidence"]  = roundf(cnn.confidence * 1000) / 1000.0f;
    t["fault_class"] = CLASS_NAMES[cnn.predicted_class];
    t["anomaly"]     = (cnn.predicted_class != IDX_NORMAL);
    t["version"]     = "2.0.0";

    // Vecteur complet des scores, dans l'ordre alphabetique de CLASS_NAMES
    // (ball, inner_race, normal, outer_race) : le dashboard peut ainsi
    // afficher les 4 classes actives au lieu de la seule gagnante.
    JsonArray probs = t.createNestedArray("probs");
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs.add(roundf(cnn.scores[i] * 1000) / 1000.0f);
    }

    JsonObject p = doc.createNestedObject("performance");
    p["cnn1d_ms"]      = cnn.latency_ms;
    p["arena_used_kb"] = cnn.arena_used / 1024;
    p["ram_free_kb"]   = ESP.getFreeHeap()  / 1024;
    p["psram_free_kb"] = ESP.getFreePsram() / 1024;

#if ENABLE_MLP
    JsonObject b = doc.createNestedObject("benchmark");
    b["mlp_class"]      = CLASS_NAMES[mlp.predicted_class];
    b["mlp_confidence"] = roundf(mlp.confidence * 1000) / 1000.0f;
    b["mlp_ms"]         = mlp.latency_ms;
    b["mlp_arena_kb"]   = mlp.arena_used / 1024;
#endif

    // Verite terrain : permet de calculer une accuracy et une matrice de
    // confusion REELLES on-device par simple requete Flux sur InfluxDB.
    doc["sim_class"] = CLASS_NAMES[sim_class];
    doc["correct"]   = (cnn.predicted_class == sim_class);

    char out[768];
    size_t n = serializeJson(doc, out, sizeof(out));
    if (n == 0 || n >= sizeof(out) - 1) {
        Serial.println("[MQTT] Payload tronque — publication annulee.");
        return;
    }

    if (mqtt_client.publish(MQTT_TOPIC, out)) {
        Serial.printf("[MQTT] Publie (%u octets) sur %s\n", (unsigned)n, MQTT_TOPIC);
    } else {
        Serial.println("[MQTT] Echec de publication.");
    }
}

// ============================================================================
//  Reseau
// ============================================================================
void setup_wifi() {
    Serial.printf("[WiFi] Connexion a %s...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts++ < 20) {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connecte — IP: %s | RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        Serial.println("\n[WiFi] Echec de connexion — mode autonome.");
    }
}

void setup_time() {
    if (WiFi.status() != WL_CONNECTED) return;
    Serial.print("[NTP] Synchronisation de l'horloge");
    configTime(0, 0, "pool.ntp.org", "time.google.com");

    int tries = 0;
    while (time(nullptr) < 1700000000 && tries++ < 20) {
        delay(500);
        Serial.print(".");
    }

    time_t now = time(nullptr);
    if (now > 1700000000) {
        Serial.printf("\n[NTP] Horloge synchronisee — epoch=%lu\n", (unsigned long)now);
    } else {
        Serial.println("\n[NTP] Echec — repli sur l'uptime (derive non calculable).");
    }
}

void setup_mqtt() {
    // Dette technique assumee, corrigee post-soutenance :
    // remplacer par setCACert(CA_CERT_PEM) + certificat serveur avec SAN valide.
    wifi_secure.setInsecure();
    mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt_client.setBufferSize(1024);

    String client_id = "esp32s3_pfe_" + String(esp_random() % 1000);
    if (mqtt_client.connect(client_id.c_str(), MQTT_USER, MQTT_PASS)) {
        Serial.println("[MQTT] Connexion MQTTS etablie (TLS 1.2).");
    } else {
        Serial.printf("[MQTT] Echec MQTTS — code=%d\n", mqtt_client.state());
    }
}

// ============================================================================
//  Setup / Loop
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(8000);   // laisse le temps d'ouvrir le moniteur serie pour la capture

    Serial.println("\n========================================");
    Serial.println("  ESP32-S3 TinyML — PredictFlow");
#if ENABLE_MLP
    Serial.println("  Mode BENCHMARK : CNN 1D + MLP");
#else
    Serial.println("  Mode PRODUCTION : CNN 1D (modele retenu)");
#endif
    Serial.println("========================================");

    print_hardware_report();

    // Doit preceder init_tflm() : le modele actif peut provenir du
    // systeme de fichiers plutot que de la memoire programme.
    ota_fs_begin();

    if (!init_tflm()) {
        Serial.println("[FATAL] Arret du systeme pour eviter le bootloop.");
        while (1) { delay(1000); }
    }

    setup_wifi();
    setup_time();
    setup_mqtt();
}

void loop() {
    if (WiFi.status() == WL_CONNECTED && !mqtt_client.connected()) {
        setup_mqtt();
    }
    mqtt_client.loop();

    // Interrogation du serveur de distribution. Placee avant le cycle
    // d'inference : un redemarrage ne peut pas couper une publication.
    static uint32_t last_ota_check = 0;
    if (millis() - last_ota_check > 60000UL) {
        last_ota_check = millis();
        ota_check_and_update(mqtt_client);
    }

    static int sim_class     = 0;
    static int cycle_counter = 0;

    Serial.printf("\n[SIM] Fenetre CWRU generee : %s\n", CLASS_NAMES[sim_class]);
    generate_cwru_window(signal_buffer, sim_class);

    BenchmarkResult cnn_res = run_inference(interp_cnn, "CNN1D", signal_buffer, false);

#if ENABLE_MLP
    BenchmarkResult mlp_res = run_inference(interp_mlp, "MLP", signal_buffer, true);
    publish_telemetry(cnn_res, signal_buffer, sim_class, mlp_res);
#else
    publish_telemetry(cnn_res, signal_buffer, sim_class);
#endif

    Serial.printf("[EVAL] attendu=%-12s predit=%-12s -> %s\n",
                  CLASS_NAMES[sim_class],
                  CLASS_NAMES[cnn_res.predicted_class],
                  (cnn_res.predicted_class == sim_class) ? "OK" : "ERREUR");

    if (++cycle_counter % INFERENCES_PER_CLASS == 0) {
        sim_class = (sim_class + 1) % NUM_CLASSES;
    }

    delay(5000);
}

