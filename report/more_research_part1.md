# More Research Part 1 – Text-to-Speech (TTS)

*Ngày cập nhật: 2025-12-03*

## 1. Bối cảnh & mục tiêu
- TTS cho phép biến văn bản thành tín hiệu âm thanh tự nhiên, được dùng trong trợ lý ảo, đọc báo tự động, sản xuất nội dung đa phương tiện và hỗ trợ người khiếm thị.
- Tuần 12 yêu cầu nắm bức tranh tổng quan: diễn tiến nghiên cứu, ưu/nhược điểm của từng hướng và cách xây dựng pipeline giảm thiểu hạn chế.

## 2. Các hướng tiếp cận chính
### 2.1 Level 1 – Luật / formant-based
- **Mô tả**: Dựa vào luật ngữ âm (diphthong, formant) + bộ tổng hợp như Klatt, Festival.
- **Ưu điểm**: 
  - Rất nhẹ, chạy real-time trên thiết bị yếu.
  - Dễ hỗ trợ đa ngôn ngữ nếu có bộ luật và từ điển phiên âm.
- **Nhược điểm**:
  - Giọng đọc đơn điệu, thiếu cảm xúc.
  - Khó mở rộng sang các phong cách nói tự nhiên.

### 2.2 Level 2 – Deep Learning pipeline cho từng speaker
- **Mô tả**: Tacotron/Tacotron2, FastSpeech, VITS,… chuyển văn bản → mel-spectrogram → vocoder (WaveGlow, HiFi-GAN). Có thể fine-tune per-speaker trên vài chục phút dữ liệu.
- **Ưu điểm**:
  - Độ tự nhiên cao, điều khiển prosody tốt.
  - Tận dụng GPU/TPU để huấn luyện nhanh, inference ổn định.
- **Nhược điểm**:
  - Cần dữ liệu lớn & sạch cho mỗi ngôn ngữ/giọng.
  - Pipeline gồm nhiều bước (text normalization → phoneme → acoustic model → vocoder), khó tối ưu end-to-end.

### 2.3 Level 3 – Few-shot / zero-shot multi-speaker
- **Mô tả**: Mô hình diffusion, VALL-E, Bark, XTTS-2, Voicebox… có khả năng clone giọng với vài giây audio.
- **Ưu điểm**:
  - Triển khai nhanh cho người dùng cuối chỉ với 3–10 giây mẫu.
  - Có thể học cảm xúc, accent, code-switching.
- **Nhược điểm**:
  - Mô hình lớn, inference tốn VRAM/compute.
  - Rủi ro đạo đức lớn (deepfake), phải kiểm soát watermark, consent.

## 3. So sánh nhanh

| Hướng | Chi phí huấn luyện | Dữ liệu cần | Tính tự nhiên | Đa ngôn ngữ | Dễ dùng | Use case đề xuất |
| --- | --- | --- | --- | --- | --- | --- |
| Luật/Formant | Rất thấp | Từ điển IPA + luật ngữ âm | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Trợ năng offline, thiết bị IoT. |
| Tacotron/FastSpeech (per speaker) | Vừa | Vài giờ ghi âm sạch/giọng | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Trợ lý doanh nghiệp, IVR có giọng thương hiệu. |
| Few-shot diffusion / VALL-E | Cao | 3–10 giây tham chiếu + tập base lớn | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ ở user cuối, khó ở backend | Nội dung sáng tạo, dubbing đa ngôn ngữ, demo R&D. |

(⭐ càng nhiều → càng tốt)

## 4. Chiến lược pipeline giảm nhược điểm
1. **Chuẩn hóa text & ngữ âm**: kết hợp G2P (grapheme-to-phoneme) để giảm lỗi phát âm; lưu cache lexicon cho từ ngoại lai.
2. **Modular hóa acoustic model & vocoder**: cho phép tráo HiFi-GAN ↔ WaveRNN tùy hạn chế tính toán.
3. **Speaker adaptation**:
   - Sử dụng pre-trained multi-speaker backbone (VD: FastSpeech2 + speaker embedding) rồi fine-tune bằng LoRA hoặc adapter để giảm dữ liệu cần thu.
   - Distillation sang phiên bản nhỏ (Edge TTS) để deploy trên thiết bị yếu.
4. **Data strategy**:
   - Augment prosody bằng perturb F0/duration, thêm noise nhẹ để tăng robust.
   - Với đa ngôn ngữ, dùng phoneme chung (IPA) + language embedding để tránh train model riêng cho từng ngôn ngữ.
5. **Emotion & style control**: thêm variance adaptor (FastSpeech2), hoặc prompt-based control (e.g., "happy", "narration").
6. **Safety & watermark**:
   - Nhúng watermark vào miền tần số hoặc embedding (SONIC, SynthID).
   - Lưu metadata (speaker consent, license) + kiểm tra nội dung đầu vào.
7. **Serving**:
   - Dùng ONNX/TensorRT để tối ưu inference cho Level 2.
   - Với Level 3 diffusion, áp dụng caching, chunked generation, hoặc decoder song song để giảm latency.

## 5. Xu hướng nghiên cứu nổi bật (2024–2025)
- **Diffusion + consistency models**: giúp tạo spectrogram mượt và giảm số bước sampling (CLAPSpeech, StyleTTS2).
- **Large Audio Models (LAM)**: hợp nhất ASR, TTS, voice conversion trong một framework (Bark, GPT-SoVITS, SeamlessM4T v2).
- **Speech tokenization**: sử dụng codec (EnCodec, SoundStream) để biểu diễn âm thanh thành discrete tokens, cho phép TTS dựa trên language models.
- **Controllability**: kết hợp text prompt + prosody prompt + emotion embedding.
- **Responsible AI**: watermark bắt buộc, nhận diện giọng tổng hợp, lưu trữ log truy vết.


