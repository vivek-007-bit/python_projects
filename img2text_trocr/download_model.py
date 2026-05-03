from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_name = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

processor.save_pretrained("./trocr_model")
model.save_pretrained("./trocr_model")

print("Model saved locally!")