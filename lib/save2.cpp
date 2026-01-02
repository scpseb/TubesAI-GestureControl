/* Includes ---------------------------------------------------------------- */
#include <string.h>
#include <WiFi.h>
// Image Classification
#include <GestureControl_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
// Camera
#include "esp_camera.h"
// BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
// FreeRTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

// PIR
#define PIR_PIN 9

// Define UUID BLE
#define SERVICE_UUID        "0001046d-9d7a-4cc4-8cd9-e385bc31104d"
#define CHARACTERISTIC_UUID "babc1b75-f950-4e74-8248-5e984a9efb09"

#define CAMERA_MODEL_XIAO_ESP32S3

#if defined(CAMERA_MODEL_XIAO_ESP32S3)
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39

#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13
#define RX1_PIN          44
#define TX1_PIN          43

#else
#error "Camera model not selected"
#endif

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS           320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS           240
#define EI_CAMERA_FRAME_BYTE_SIZE                 3

// Threshold prediction
#define CONF_THRESHOLD 0.70f
#define STABLE_COUNT   3

// PIR timing
#define PIR_HOLD_MS 5000

// Task priorities (higher number = higher priority)
#define PIR_TASK_PRIORITY        3
#define INFERENCE_TASK_PRIORITY  2
#define BLE_TASK_PRIORITY        1

// Task stack sizes
#define PIR_TASK_STACK_SIZE        2048
#define INFERENCE_TASK_STACK_SIZE  8192
#define BLE_TASK_STACK_SIZE        4096

// Queue sizes
#define BLE_QUEUE_SIZE  5

/* Global variables -------------------------------------------------------- */
// Camera
static bool is_initialised = false;
uint8_t *snapshot_buf;

// Inference state (protected by mutex)
static int last_prediction = -1;
static int stable_counter = 0;
static int last_sent_code = -1;

// PIR state
static volatile bool motion_active = false;
static volatile uint32_t last_pir_high_ms = 0;

// BLE
BLECharacteristic *pCharacteristic = nullptr;
volatile bool deviceConnected = false;

// FreeRTOS handles
TaskHandle_t pirTaskHandle = NULL;
TaskHandle_t inferenceTaskHandle = NULL;
TaskHandle_t bleTaskHandle = NULL;
QueueHandle_t bleQueue = NULL;
SemaphoreHandle_t cameraMutex = NULL;
SemaphoreHandle_t inferenceMutex = NULL;

// BLE message structure
typedef struct {
    uint8_t code;
} ble_message_t;

/* Camera configuration ---------------------------------------------------- */
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 10000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,

    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

/* Function declarations --------------------------------------------------- */
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);
static int label_to_code(int pred_idx);

// Task functions
void pirTask(void *pvParameters);
void inferenceTask(void *pvParameters);
void bleTask(void *pvParameters);

/* BLE Callbacks ----------------------------------------------------------- */
class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("BLE: Client connected");
    }
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("BLE: Client disconnected");
        pServer->getAdvertising()->start();
    }
};

/* Setup ------------------------------------------------------------------- */
void setup()
{
    WiFi.mode(WIFI_OFF);
    pinMode(PIR_PIN, INPUT);
    
    Serial.begin(115200);
    setCpuFrequencyMhz(160);
    
    while (!Serial);
    Serial.println("===========================================");
    Serial.println("Gesture Control with FreeRTOS - Starting...");
    Serial.println("===========================================");

    // Create mutexes
    cameraMutex = xSemaphoreCreateMutex();
    inferenceMutex = xSemaphoreCreateMutex();
    
    if (cameraMutex == NULL || inferenceMutex == NULL) {
        Serial.println("ERR: Failed to create mutexes!");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    // Create BLE queue
    bleQueue = xQueueCreate(BLE_QUEUE_SIZE, sizeof(ble_message_t));
    if (bleQueue == NULL) {
        Serial.println("ERR: Failed to create BLE queue!");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    // Initialize BLE
    BLEDevice::init("Gesture-XIAO");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharacteristic->addDescriptor(new BLE2902());
    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->start();
    Serial.println("BLE: Advertising started");

    // Initialize camera
    Serial.println("Initializing camera...");
    if (ei_camera_init() == false) {
        Serial.println("ERR: Failed to initialize Camera!");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }
    Serial.println("Camera initialized successfully");

    // Allocate snapshot buffer in PSRAM
    snapshot_buf = (uint8_t*)heap_caps_malloc(
        EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (!snapshot_buf) {
        Serial.println("ERR: Failed to allocate snapshot buffer!");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }
    Serial.println("Snapshot buffer allocated");

    // Create FreeRTOS tasks
    Serial.println("Creating RTOS tasks...");

    // PIR monitoring task - runs on Core 0
    BaseType_t result = xTaskCreatePinnedToCore(
        pirTask,
        "PIR_Task",
        PIR_TASK_STACK_SIZE,
        NULL,
        PIR_TASK_PRIORITY,
        &pirTaskHandle,
        0  // Core 0
    );
    if (result != pdPASS) {
        Serial.println("ERR: Failed to create PIR task!");
    }

    // Inference task - runs on Core 1 (main processing core)
    result = xTaskCreatePinnedToCore(
        inferenceTask,
        "Inference_Task",
        INFERENCE_TASK_STACK_SIZE,
        NULL,
        INFERENCE_TASK_PRIORITY,
        &inferenceTaskHandle,
        1  // Core 1
    );
    if (result != pdPASS) {
        Serial.println("ERR: Failed to create Inference task!");
    }

    // BLE transmission task - runs on Core 0
    result = xTaskCreatePinnedToCore(
        bleTask,
        "BLE_Task",
        BLE_TASK_STACK_SIZE,
        NULL,
        BLE_TASK_PRIORITY,
        &bleTaskHandle,
        0  // Core 0
    );
    if (result != pdPASS) {
        Serial.println("ERR: Failed to create BLE task!");
    }

    Serial.println("===========================================");
    Serial.println("All tasks created. System running...");
    Serial.println("===========================================");
}

/* Loop - empty since we use RTOS tasks ------------------------------------ */
void loop()
{
    // Main loop is not used - all work is done in RTOS tasks
    vTaskDelay(pdMS_TO_TICKS(1000));
}

/* PIR Monitoring Task ----------------------------------------------------- */
void pirTask(void *pvParameters)
{
    bool printed_idle = false;
    
    Serial.println("[PIR Task] Started");
    
    while (1) {
        bool pir_high = (digitalRead(PIR_PIN) == HIGH);
        
        if (pir_high) {
            last_pir_high_ms = millis();
        }
        
        bool active_window = (millis() - last_pir_high_ms) <= PIR_HOLD_MS;
        
        if (active_window) {
            if (!motion_active) {
                motion_active = true;
                printed_idle = false;
                Serial.println("[PIR Task] Motion ACTIVE -> Resume inference");
                
                // Notify inference task
                if (inferenceTaskHandle != NULL) {
                    xTaskNotifyGive(inferenceTaskHandle);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(50));  // Check every 50ms when active
        } else {
            if (motion_active) {
                motion_active = false;
                
                // Reset inference state
                if (xSemaphoreTake(inferenceMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                    stable_counter = 0;
                    last_prediction = -1;
                    xSemaphoreGive(inferenceMutex);
                }
            }
            
            if (!printed_idle) {
                printed_idle = true;
                Serial.println("[PIR Task] Motion IDLE -> Inference paused");
            }
            
            vTaskDelay(pdMS_TO_TICKS(200));  // Check every 200ms when idle (power saving)
        }
    }
}

/* Inference Task ---------------------------------------------------------- */
void inferenceTask(void *pvParameters)
{
    Serial.println("[Inference Task] Started");
    
    ei_impulse_result_t result = { 0 };
    
    while (1) {
        // Wait for motion or check periodically
        if (!motion_active) {
            // Wait for notification from PIR task (with timeout)
            ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(500));
            continue;
        }
        
        // Take camera mutex
        if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(1000)) != pdTRUE) {
            Serial.println("[Inference Task] Failed to acquire camera mutex");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        
        // Capture image
        bool capture_ok = ei_camera_capture(
            EI_CLASSIFIER_INPUT_WIDTH, 
            EI_CLASSIFIER_INPUT_HEIGHT, 
            snapshot_buf
        );
        
        xSemaphoreGive(cameraMutex);
        
        if (!capture_ok) {
            Serial.println("[Inference Task] Capture failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        
        // Prepare signal
        ei::signal_t signal;
        signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
        signal.get_data = &ei_camera_get_data;
        
        // Run classifier
        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        if (err != EI_IMPULSE_OK) {
            Serial.printf("[Inference Task] Classifier error: %d\n", err);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        
        // Find top prediction
        int current_prediction = -1;
        float max_value = 0.0f;
        
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            float v = result.classification[i].value;
            if (v > max_value) {
                max_value = v;
                current_prediction = (int)i;
            }
        }
        
        // Temporal smoothing with mutex protection
        if (xSemaphoreTake(inferenceMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            if (max_value >= CONF_THRESHOLD) {
                if (current_prediction == last_prediction) {
                    stable_counter++;
                } else {
                    last_prediction = current_prediction;
                    stable_counter = 1;
                }
            } else {
                last_prediction = -1;
                stable_counter = 0;
            }
            
            // Check if prediction is stable
            if (stable_counter >= STABLE_COUNT) {
                int code = label_to_code(current_prediction);
                
                Serial.printf("[Inference Task] FINAL => %s | code=%d | conf=%.2f\n",
                    ei_classifier_inferencing_categories[current_prediction],
                    code,
                    max_value
                );
                
                // Send to BLE queue if valid and different from last sent
                if (code >= 0 && code != last_sent_code) {
                    ble_message_t msg;
                    msg.code = (uint8_t)code;
                    
                    if (xQueueSend(bleQueue, &msg, pdMS_TO_TICKS(10)) == pdTRUE) {
                        last_sent_code = code;
                    }
                }
                
                // Reset for next detection
                stable_counter = 0;
                last_prediction = -1;
            }
            
            xSemaphoreGive(inferenceMutex);
        }
        
        // Inference interval (power saving)
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* BLE Transmission Task --------------------------------------------------- */
void bleTask(void *pvParameters)
{
    Serial.println("[BLE Task] Started");
    
    ble_message_t msg;
    
    while (1) {
        // Wait for message in queue
        if (xQueueReceive(bleQueue, &msg, pdMS_TO_TICKS(100)) == pdTRUE) {
            if (deviceConnected && pCharacteristic != nullptr) {
                pCharacteristic->setValue(&msg.code, 1);
                pCharacteristic->notify();
                Serial.printf("[BLE Task] Sent: %d\n", msg.code);
            } else {
                Serial.println("[BLE Task] Not connected, message dropped");
            }
        }
        
        // Small delay to prevent tight loop
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* Camera Functions -------------------------------------------------------- */
bool ei_camera_init(void)
{
    if (is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, 0);
    }

    is_initialised = true;
    return true;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf)
{
    bool do_resize = false;

    if (!is_initialised) {
        Serial.println("ERR: Camera is not initialized");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        Serial.println("Conversion failed");
        return false;
    }

    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) ||
        (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
            snapshot_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf,
            img_width,
            img_height
        );
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr)
{
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        // Swap BGR to RGB
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + 
                              (snapshot_buf[pixel_ix + 1] << 8) + 
                              snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    return 0;
}

static int label_to_code(int pred_idx)
{
    const char* lab = ei_classifier_inferencing_categories[pred_idx];

    if (strcmp(lab, "0 Jari") == 0) return 0;
    if (strcmp(lab, "1 Jari") == 0) return 1;
    if (strcmp(lab, "2 Jari") == 0) return 2;

    return -1;
}

void ei_camera_deinit(void)
{
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        Serial.printf("Camera deinit failed (0x%x)\n", err);
    }
    is_initialised = false;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif