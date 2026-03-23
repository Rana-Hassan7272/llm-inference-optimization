Model Details
Model: TinyLlama-1.1B-Chat-v1.0
Source: Hugging Face
Precision: FP16
Device: NVIDIA T4 (Google Colab)
Framework: PyTorch + Transformers
Experiment Setup
Prompt: "Explain machine learning in simple terms"
Max new tokens: 50
Batch size: 1
KV Cache: Enabled (default in transformers)
Execution environment: Google Colab (GPU enabled)
Performance Metrics
Total Inference Time: 1.18 seconds
Time per token: ~0.02–0.03 sec (approx, based on output length)
Tokens generated: ~50
Estimated Throughput: ~40–50 tokens/sec
Output Quality (Manual Evaluation)
Score: 4.5 / 5
Observations:
Response is clear and understandable
Suitable for beginner-level explanation
No hallucinations observed
Slight verbosity but acceptable
Observations
Model loads quickly (~2–3 seconds after download)
GPU acceleration significantly improves generation speed
TinyLlama performs surprisingly well for a 1.1B model
No memory issues encountered on T4 GPU