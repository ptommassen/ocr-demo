USER_LOCAL_PATH:=$(LOCAL_PATH)
LOCAL_PATH:=$(subst ?,,$(firstword ?$(subst \, ,$(subst /, ,$(call my-dir)))))

ifeq ($(TESS_$(TARGET_ARCH_ABI)_ALREADY_INCLUDED),)

TESSERACT_THIS_DIR:=$(patsubst $(LOCAL_PATH)\\%,%,$(patsubst $(LOCAL_PATH)/%,%,$(call my-dir)))

include $(CLEAR_VARS)
LOCAL_MODULE := lept
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/liblept.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := tess
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libtess.so
include $(PREBUILT_SHARED_LIBRARY)

MY_C_INCLUDES += $(TESSERACT_THIS_DIR)/include
MY_SHARED_LIBRARIES += lept tess


TESS_$(TARGET_ARCH_ABI)_ALREADY_INCLUDED:=on

endif

LOCAL_PATH:=$(USER_LOCAL_PATH)