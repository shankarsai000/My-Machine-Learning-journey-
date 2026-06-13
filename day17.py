# ==========================================
# DAY 17: PROBABILITY FOR MACHINE LEARNING
# ==========================================
import numpy as np
# ------------------------------------------
# 1. BASIC PROBABILITY
# ------------------------------------------
# Probability = Favorable Outcomes / Total Outcomes

total_outcomes = 6
favorable_outcomes = 1

probability = favorable_outcomes / total_outcomes

print("Probability of rolling a 1:")
print(probability)

# P(1) = 1/6
# ------------------------------------------
# 2. SAMPLE SPACE
# ------------------------------------------
# All possible outcomes

sample_space = [1, 2, 3, 4, 5, 6]
print("\nSample Space:")
print(sample_space)
# ------------------------------------------
# 3. EVENT
# ------------------------------------------
# Event = Getting an even number

event = [2, 4, 6]

prob_even = len(event) / len(sample_space)

print("\nProbability of Even Number:")
print(prob_even)

# P(Even) = 3/6 = 0.5
# ------------------------------------------
# 4. COMPLEMENT RULE
# ------------------------------------------
# P(Not A) = 1 - P(A)

prob_not_even = 1 - prob_even

print("\nProbability of NOT Even:")
print(prob_not_even)
# ------------------------------------------
# 5. INDEPENDENT EVENTS
# ------------------------------------------
# Coin Toss and Dice Roll

p_head = 1/2
p_six = 1/6

joint_probability = p_head * p_six

print("\nIndependent Events:")
print("P(Head AND Six):", joint_probability)
# ------------------------------------------
# 6. DEPENDENT EVENTS
# ------------------------------------------
# Drawing cards without replacement

first_card_ace = 4/52
second_card_ace = 3/51

dependent_probability = first_card_ace * second_card_ace

print("\nDependent Events:")
print("Two Aces:", dependent_probability)
# ------------------------------------------
# 7. CONDITIONAL PROBABILITY
# ------------------------------------------
# P(A|B) = P(A and B) / P(B)

p_rain = 0.3
p_umbrella_given_rain = 0.9

print("\nConditional Probability Example:")
print("P(Umbrella | Rain) =", p_umbrella_given_rain)
# ------------------------------------------
# 8. BAYESIAN THINKING
# ------------------------------------------
# Prior Belief -> Evidence -> Updated Belief

prior = 0.01      # Disease rate
likelihood = 0.95 # Positive if disease
false_positive = 0.05

print("\nBayesian Thinking:")
print("Prior Probability:", prior)
print("Likelihood:", likelihood)
# ------------------------------------------
# 9. BAYES THEOREM EXAMPLE
# ------------------------------------------

P_D = 0.01
P_Pos_given_D = 0.95
P_Pos_given_NotD = 0.05

P_NotD = 1 - P_D

P_Pos = (
    P_Pos_given_D * P_D +
    P_Pos_given_NotD * P_NotD
)

P_D_given_Pos = (
    P_Pos_given_D * P_D
) / P_Pos

print("\nBayes Result:")
print("Probability of Disease Given Positive Test:")
print(round(P_D_given_Pos, 4))
# ------------------------------------------
# 10. COIN TOSS SIMULATION
# ------------------------------------------

coin_tosses = np.random.choice(
    ["H", "T"],
    size=1000
)

heads = np.sum(coin_tosses == "H")

print("\nCoin Toss Simulation:")
print("Heads Probability:",
      heads / 1000)
# ------------------------------------------
# 11. DICE ROLL SIMULATION
# ------------------------------------------

dice = np.random.randint(
    1,
    7,
    10000
)

probability_six = np.sum(dice == 6) / len(dice)

print("\nDice Simulation:")
print("P(6) ≈", probability_six)
# ------------------------------------------
# 12. REAL ML EXAMPLE
# ------------------------------------------
# Spam Detection Thinking

spam_probability = 0.20

contains_free_given_spam = 0.80

contains_free_given_not_spam = 0.10

print("\nSpam Detection Idea:")
print("Prior Spam Probability:",
      spam_probability)

print("Evidence: Word 'FREE'")
# ------------------------------------------
# 13. WHY PROBABILITY MATTERS
# ------------------------------------------

print("\nML Insight:")
print("ML rarely works with certainty.")
print("Models work with probabilities.")
print("Predictions are confidence estimates.")
print("Probability powers Classification.")
print("Probability powers Naive Bayes.")
print("Probability powers Modern AI.")