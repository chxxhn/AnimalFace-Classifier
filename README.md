# AnimalFace-Classifier🐶🐱🐹 

## 📌 프로젝트 개요
본 프로젝트는 **머신러닝 모델을 이용하여 얼굴 이미지를 분석하고, 닮은 동물 상(고양이, 햄스터, 강아지)을 분류하는 Android 애플리케이션**입니다.  
사용자는 갤러리에서 이미지를 업로드하면, **TensorFlow Lite 기반의 머신러닝 모델이 얼굴을 분석하여 가장 닮은 동물 유형을 예측**합니다.

---

## 🛠 주요 기능
### 1️⃣ **이미지 선택 및 업로드**
- **갤러리 API 연동**: 사용자가 갤러리에서 이미지를 선택
- **선택한 이미지 URI 가져오기**: `Intent.ACTION_PICK`을 사용하여 이미지 선택 후 `Bitmap` 변환

### 2️⃣ **이미지 처리**
- **비트맵 변환 및 전처리**: 업로드한 이미지를 머신러닝 모델 입력 형식(`28x28 grayscale`)으로 변환
- **픽셀 정규화**: RGB 값을 **0~1 범위로 변환**하여 모델 입력값 생성

### 3️⃣ **머신러닝을 통한 동물상 분석**
- **TensorFlow Lite 모델을 이용한 분석**
  - CNN(Convolutional Neural Network) 기반 **이미지 분류 모델** 적용
  - `tflite` 모델을 사용하여 **이미지를 실시간 분석**
- **모델 훈련 및 적용**
  - 학습 데이터 전처리 및 CNN 모델 훈련 (`TensorFlow`)
  - `TFLiteConverter`를 사용하여 `.tflite` 파일 변환 및 안드로이드 앱 통합

### 4️⃣ **분석 결과 표시**
- **예측 결과 출력**: 분석된 **닮은 동물 유형(고양이, 강아지, 햄스터)**을 확률과 함께 화면에 표시
- **해당 동물 이미지 출력**: 예측 결과에 따라 동물 사진 표시
- **사용자 인터랙션**
  - `다시하기` 버튼: 다른 이미지 분석 가능
  - `종료하기` 버튼: 앱 종료 또는 초기 화면 복귀

---

## 🔧 사용된 기술 스택
- **언어:** Kotlin
- **UI 구성:** `ImageView`, `Button`, `LinearLayout`, `TextView`
- **머신러닝 및 모델 적용**
  - **TensorFlow Lite** → `.tflite` 모델 변환 및 실행
  - **CNN (Convolutional Neural Network)** → 이미지 분류 모델 학습
  - **Interpreter API** → TFLite 모델을 로드하고 분석 실행
  - **ByteBuffer 변환** → 이미지 데이터를 모델 입력 형식으로 변환
- **이미지 처리 및 저장**
  - `Bitmap.createScaledBitmap()` → 입력 이미지를 `28x28` 크기로 변환
  - `getPixel()` → RGB 값을 추출 후 Grayscale 변환
  - `Intent.ACTION_PICK` → 갤러리에서 이미지 가져오기
- **파일 관리 및 권한 처리**
  - `WRITE_EXTERNAL_STORAGE`, `READ_EXTERNAL_STORAGE` → 갤러리 접근 권한
  - `FileInputStream`, `FileOutputStream` → 모델 파일 저장 및 로드

---

## 🎯 프로젝트 진행 과정
1. **이미지 데이터셋 전처리 및 모델 학습**
   - TensorFlow를 사용하여 **CNN 모델** 구축 및 훈련
   - 학습된 모델을 `.tflite` 형식으로 변환

2. **안드로이드 앱 구현**
   - 사용자가 이미지를 업로드할 수 있도록 **갤러리 API 연동**
   - `Bitmap`을 **TensorFlow Lite 입력 형식**으로 변환하여 모델에 적용

3. **TensorFlow Lite 통합 및 결과 출력**
   - `Interpreter.run()`을 사용하여 모델을 실행하고 **예측 결과 도출**
   - UI를 통해 **예측된 동물 유형과 확률을 표시**

---

## 🏆 기대 효과 및 개선 방향
✅ **기대 효과**
- 머신러닝을 활용하여 **이미지를 분석하고 재미있는 결과(동물상 예측)** 제공
- TensorFlow Lite를 사용하여 **모바일 환경에서 경량화된 AI 모델 적용**

❌ **향후 개선 방향**
- **카메라 기능 추가**: 실시간 사진 촬영 후 분석 가능하도록 기능 확장
- **추가 동물 카테고리 지원**: 현재 고양이, 강아지, 햄스터 외에도 다양한 동물상 예측 가능하도록 데이터셋 확장


