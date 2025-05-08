#pragma once

#include <stdint.h>

#define MAX_LONG_NAME_LEN 56
#define MAX_LONG_NAME_FRAMES 8
#define MAX_MESSAGE_LEN 190
#define MAX_MESSAGE_FRAMES 32
#define MAX_AUDIO_SERVICES 8
#define MAX_DATA_SERVICES 16
#define NUM_PARAMETERS 13
#define MAX_UNIVERSAL_SHORT_NAME_LEN 12
#define MAX_UNIVERSAL_SHORT_NAME_FRAMES 2
#define MAX_SLOGAN_LEN 95
#define MAX_SLOGAN_FRAMES 16
#define MAX_ALERT_LEN 381
#define MAX_ALERT_FRAMES 64
#define MAX_ALERT_CNT_LEN 63
#define MAX_ALERT_LOCATIONS 31

typedef struct
{
    int access;
    int type;
    int sound_exp;
} asd_t;

typedef struct
{
    int access;
    int type;
    int mime_type;
} dsd_t;

typedef enum
{
    ENCODING_ISO_8859_1 = 0,
    ENCODING_UCS_2 = 4
} encoding_t;

typedef struct
{
    struct input_t *input;

    char country_code[3];
    int fcc_facility_id;

    char short_name[8];

    char long_name[MAX_LONG_NAME_LEN + 1];
    uint8_t long_name_have_frame[MAX_LONG_NAME_FRAMES];
    int long_name_seq;
    int long_name_displayed;

    float latitude;
    float longitude;
    int altitude;

    char message[MAX_MESSAGE_LEN + 1];
    uint8_t message_have_frame[MAX_MESSAGE_FRAMES];
    int message_seq;
    int message_priority;
    encoding_t message_encoding;
    int message_len;
    unsigned int message_checksum;
    int message_displayed;

    asd_t audio_services[MAX_AUDIO_SERVICES];
    dsd_t data_services[MAX_DATA_SERVICES];

    int parameters[NUM_PARAMETERS];

    char universal_short_name[MAX_UNIVERSAL_SHORT_NAME_LEN + 1];
    char universal_short_name_final[MAX_UNIVERSAL_SHORT_NAME_LEN + 4];
    uint8_t universal_short_name_have_frame[MAX_UNIVERSAL_SHORT_NAME_FRAMES];
    encoding_t universal_short_name_encoding;
    int universal_short_name_append;
    int universal_short_name_len;
    int universal_short_name_displayed;

    char slogan[MAX_SLOGAN_LEN + 1];
    uint8_t slogan_have_frame[MAX_SLOGAN_FRAMES];
    encoding_t slogan_encoding;
    int slogan_len;
    int slogan_displayed;

    char alert[MAX_ALERT_LEN + 1];
    uint8_t alert_have_frame[MAX_ALERT_FRAMES];
    int alert_seq;
    encoding_t alert_encoding;
    int alert_len;
    int alert_crc;
    int alert_cnt_len;
    int alert_displayed;
    int alert_timeout;
} pids_t;

void pids_frame_push(pids_t *st, uint8_t *bits);
void pids_init(pids_t *st, struct input_t *input);
