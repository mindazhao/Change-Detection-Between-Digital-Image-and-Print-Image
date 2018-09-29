
#ifndef CALLBACK_H_FILE
#define CALLBACK_H_FILE
#ifdef CPPLIB
#define CPPLIB extern "C" _declspec(dllimport) 
#else
#define CPPLIB extern "C" _declspec(dllexport) 
#endif

#include "stdafx.h"

typedef bool (CALLBACK *Receive)(int p);
Receive m_RecInfoCall;  //�ظ���Ϣ�Ļص�����
extern "C"__declspec(dllexport)BOOL WINAPI SetCallbackCombine(Receive InfoReceive)
{
	m_RecInfoCall = InfoReceive;
	return TRUE;
}

#endif