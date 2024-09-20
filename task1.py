from aitextgen import aitextgen


# Initialize aitextgen with default model (GPT-2 small)
ai = aitextgen()

# Fine-tune the model on the sample dataset
ai.train("sample.txt", num_steps=5000)

# Save the trained model
ai.save("trained_model")

# Generate text based on a prompt
prompt = "In the heart of the kingdom,"
generated_text = ai.generate_one(prompt)
print("Generated Text:", generated_text)
