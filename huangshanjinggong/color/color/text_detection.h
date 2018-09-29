#ifndef TEXT_DETECTION_H_
#define TEXT_DETECTION_H_
#ifdef TEXT_DETECTIONLIB
#define TEXT_DETECTIONLIB extern "C" _declspec(dllimport) 
#else
#define TEXT_DETECTIONLIB extern "C" _declspec(dllexport) 
#endif
//CPPLIB int AddCalc(int plus1, int plus2);
//You can also write like this:
//extern "C" {
//_declspec(dllexport) int AddCalc(int plus1, int plus2);
//};
#endif