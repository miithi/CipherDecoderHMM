import numpy as np
from hmmlearn import hmm

# Zodiac 408 Cipher 
ciphertext = "jmjlfmjmmjohqfpqmfcfbvtfjujtqpnvdigvonjujtopsfgvoibokmmjohxjmehbnfjouifgpssfucfbvtfnbojtuifnptvubohfspvfbobncmpgbmmtpmjmtpnfuijohhjwftnfujifnptuuisjmmjohfyqfsjfodfjujtfwfocfuufsuibohfuujohzpvsqpdltpggxjuibhsmpuifcftuqbsuvojujtjtbixifojejfxjmncfsfcspsjoqbsbejdfboebmmuifjibwfljmmfejmmcfdpnfnztmbwftjxjmmpohjwfzpvnzobnfdbvtfzpvxjmmuszuptmpxnfepxo psbupqnzdpmmfdujphpgtnmbwftgpsnzbgufsmjgffcfpsjfufofuijqkvuj"
plaintext = "ilikekillingpeoplebecauseitissomuchfunitismorefunthankillingwildgameintheforrestbecausemanisthemostdangeroueanamalofalltokillsomethinggivesmethemostthrillingexperienceitisevenbetterthangettingyourrocksoffwithagirlthebestparttofitisthaewhenidieiwillbereborninparadiceandalltheihavekilledwillbecomemyslavesiwillnotgiveyoumynamebecauseyouwilltrytosloidownoratopmycollectiogofslavesformyafterlifeebeorietemethhpiti"

# Create a mapping between symbols and states (characters)
symbols = list(set(ciphertext))  
states = list(set(plaintext))  

# Convert cipher text and plain text to indices
cipher_indices = [symbols.index(c) for c in ciphertext]
plain_indices = [states.index(p) for p in plaintext]

# Custom matrices provided
A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

pi = np.array([0, 1])  

# Function to compute accuracy
def compute_accuracy(decoded_states, true_states):
    correct = sum([1 for i in range(len(decoded_states)) if decoded_states[i] == true_states[i]])
    return correct / len(true_states) * 100

# HMM Training and Decoding with random restarts for the emission matrix B
def train_hmm_with_random_restarts(n_restarts, n_iterations, re_estimate_A=False):
    best_accuracy = 0
    best_model = None

    for restart in range(n_restarts):
        # Initialize HMM model with random emission matrix B
        model = hmm.MultinomialHMM(n_components=2, n_iter=n_iterations, tol=1e-4)
        
        # Set the fixed transition matrix A and initial state distribution pi
        model.transmat_ = A
        model.startprob_ = pi
        
        if not re_estimate_A:
            model.transmat_ = A  # Keep transition matrix fixed
        else:
            model.transmat_ = np.random.rand(2, 2)  # Random initialization if re-estimating A
        
        # Random initialization of the emission matrix B
        model.emissionprob_ = np.random.rand(2, len(symbols))

        # Train the HMM (only for B, not A if re-estimate_A is False)
        model.fit(np.array(cipher_indices).reshape(-1, 1))  # Baum-Welch for emission matrix B

        # Decode the cipher using the trained HMM
        decoded_states = model.predict(np.array(cipher_indices).reshape(-1, 1))

        # Calculate accuracy compared to the known plaintext
        accuracy = compute_accuracy(decoded_states, plain_indices)

        # Track the best result
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_accuracy

# Parameters
n_iterations = 200

# Part a: 1,000 Random Restarts
accuracy_1000 = train_hmm_with_random_restarts(n_restarts=1000, n_iterations=n_iterations, re_estimate_A=False)
print(f"Accuracy with 1,000 restarts: {accuracy_1000:.2f}%")

# Part b: 10,000 Random Restarts
accuracy_10000 = train_hmm_with_random_restarts(n_restarts=10000, n_iterations=n_iterations, re_estimate_A=False)
print(f"Accuracy with 10,000 restarts: {accuracy_10000:.2f}%")

# Part c: 100,000 Random Restarts
accuracy_100000 = train_hmm_with_random_restarts(n_restarts=100000, n_iterations=n_iterations, re_estimate_A=False)
print(f"Accuracy with 100,000 restarts: {accuracy_100000:.2f}%")
