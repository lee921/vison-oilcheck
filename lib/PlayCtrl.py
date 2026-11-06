# coding=utf-8

from ctypes import *
import sys

# 回调函数类型定义
if "linux" in sys.platform:
    fun_ctype = CFUNCTYPE
else:
    fun_ctype = WINFUNCTYPE


# 定义预览参数结构体
class FRAME_INFO(Structure):
    pass


# PlayCtrl 的 LONG 在 Linux 64 位 SDK 中仍按 4 字节 ABI 传递。
# ctypes.c_long 在该平台是 8 字节，会破坏端口号等参数的调用约定。
PLAYM4_LONG = c_int
PLAYM4_PORT = PLAYM4_LONG


FRAME_INFO._fields_ = [
    ("nWidth", c_uint32),
    ("nHeight", c_uint32),
    ("nStamp", c_uint32),
    ("nType", c_uint32),
    ("nFrameRate", c_uint32),
    ("dwFrameNum", c_uint32),
]
LPFRAME_INFO = POINTER(FRAME_INFO)


# 定义预览参数结构体
class DISPLAY_INFO_YUV(Structure):
    _fields_ = [
        ("nPort", c_uint32),  # 通道号
        ("pBuf", c_char_p),  # 返回的第一路图像数据指针
        ("nBufLen", c_uint),  # 返回的第一路图像数据大小
        ("pBuf1", c_char_p),  # 返回的第二路图像数据指针
        ("nBufLen1", c_uint),  # 返回的第二路图像数据大小
        ("pBuf2", c_char_p),  # 返回的第三路图像数据指针
        ("nBufLen2", c_uint),  # 返回的第三路图像数据大小
        ("nWidth", c_uint),  # 画面宽
        ("nHeight", c_uint),  # 画面高
        ("nStamp", c_uint),  # 时标信息，单位毫秒
        ("nType", c_uint),  # 数据类型
        ("pUser", c_void_p),  # 用户数据
        ("reserved", c_uint * 4),  # 保留,reserved[0]保存帧号,res[1]-res[3]保存全局时间
    ]


LP_DISPLAY_INFO_YUV = POINTER(DISPLAY_INFO_YUV)

# 显示回调函数
DISPLAYCBFUN = fun_ctype(
    None,
    PLAYM4_PORT,
    c_char_p,
    PLAYM4_LONG,
    PLAYM4_LONG,
    PLAYM4_LONG,
    PLAYM4_LONG,
    PLAYM4_LONG,
    PLAYM4_LONG,
)

# 解码回调函数
DECCBFUNWIN = fun_ctype(
    None,
    PLAYM4_PORT,
    POINTER(c_char),
    PLAYM4_LONG,
    POINTER(FRAME_INFO),
    c_void_p,
    c_void_p,
)

# 硬解码回调函数
DeCCBFUNWINHARD = fun_ctype(None, LP_DISPLAY_INFO_YUV)
