import math

#[-8.14405058  1.14851008  0.29156678]

def predict(cgpa, experience):
    bias = -8.14405058
    w_cgpa = 1.14851008
    w_exp = 0.29156678

    z = bias + (w_cgpa * cgpa) + (w_exp * experience)

    probability = 1 / (1 + math.exp(-z))

    label = "YES" if probability >= 0.5 else "NO"

    return probability, label

cgpa = float(input("CGPA: "))
experience = float(input("Experience: "))

prob, result = predict(cgpa, experience)

print(f"Probability of getting job: {prob:.4f}")
print(f"Prediction: {result}")