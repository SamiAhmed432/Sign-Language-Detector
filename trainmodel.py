from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}

# Determine the desired sequence length
desired_sequence_length = 19  # Adjust this to your desired length

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(desired_sequence_length):
            # Add allow_pickle=True here to handle loading pickled arrays
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)

            # Check if the loaded array is an object (which might indicate sequences of varying lengths)
            if res.ndim == 0 or res.ndim == 2:
                print(f"Warning: Skipping sequence with unexpected shape for {action}/{sequence}/{frame_num}")
                continue

            window.append(res)

        # Check if the window is empty
        if not window:
            print(f"Warning: Skipping empty sequence for {action}/{sequence}")
            continue

        # Reshape the sequence to have a fixed length
        reshaped_sequence = np.zeros((desired_sequence_length, window[0].shape[0]))
        for i, frame in enumerate(window):
            reshaped_sequence[i, :] = frame

        sequences.append(reshaped_sequence)
        labels.append(label_map[action])

# Convert to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up the model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(desired_sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile and train the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')