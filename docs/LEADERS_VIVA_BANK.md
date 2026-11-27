# Project 8: complete viva and team question bank

Use this as the group's single question-and-answer reference. Learn the short answer first, then understand the explanation underneath it. Don't claim more than the experiments show.

## 30-second project introduction

“Our project is an industrial-style wearable movement classifier using DASIG sensor data. We use acceleration and gyroscope time series from one forearm IMU and a CNN-LSTM to classify 3-second windows as Standard pick-and-place or Abrupt alarm-response movement. We keep participants separate between training, validation, and test. On the held-out DASIG participants we got 97.1% accuracy and 93.7% macro F1. We also made an Android phone browser demo that gives live predictions and saves labeled phone recordings, but those phone results are separate from the benchmark because the sensor placement and device differ.”

## Project purpose and scope

### 1. What is the objective of the project?

Build a neural sequence classifier using wearable accelerometer and gyroscope readings for an industrial-style movement recognition task. The selected DASIG classes are Standard pick-and-place and Abrupt alarm response.

### 2. Why does this fit the assigned Bosch idea?

The assigned idea asks for wearable-sensor gesture/activity recognition for factory human-machine interaction. DASIG provides industrial-style wearable IMU movement data. Our work is a small proof of concept for recognizing two movement types, not a complete factory command interface.

### 3. What does your model predict?

For each 3-second sensor window, it predicts one of two classes: Standard or Abrupt. It does not predict an alarm, machine breakdown, worker identity, or a command such as Start or Stop.

### 4. What is the input and output?

Input is a sequence with shape 150 time steps by 6 channels: acceleration X/Y/Z and gyroscope X/Y/Z. Output is a score for each of the two movement classes; the larger score is the model's predicted class.

### 5. Is this an end-to-end factory product?

No. DASIG is a controlled simulated workstation dataset. The project demonstrates a modeling pipeline and needs new real-factory data and validation before any production claim.

## Dataset and labeling

### 6. What is DASIG?

DASIG is a wearable inertial-sensor dataset of industrial-style pick-and-place trials and responses to visual/audio alarm cues. Participants wore IMUs at multiple body positions.

### 7. How many people and sensors are in the dataset?

The dataset has 60 participants and five IMUs positioned on the forearms, upper arms, and chest. Our model uses one active forearm sensor per trial, not all five devices.

### 8. Which sensor signals do you use?

Three-axis accelerometer and three-axis gyroscope, six channels total. The raw sample rate is 200 Hz; our windows are resampled to 50 Hz for a compact model and phone demonstration.

### 9. What do Standard and Abrupt mean?

Standard is a pick-and-place interval with no alarm event before the next pick cue. Abrupt is a window beginning at a green pick cue that has a red-light or buzzer event before the next green cue.

### 10. Does DASIG label every sensor row with an activity?

No. It provides synchronized event times/codes. We derive window labels by aligning those cues with the IMU stream. That makes our labels event-derived proxy labels, not dense manual annotations of exact movement onset/offset.

### 11. Why call it a proxy label?

An alarm event signals the task condition, but the person reacts after some delay. A 3-second window can contain part of the preceding motion and part of the response. The cue time is therefore only an approximation of movement label boundaries.

### 12. Why were some trials excluded?

The DASIG authors identify five trials where the expected visual-alarm response was missing. Keeping these as Abrupt examples would assign a response label to a trial without the response, so we exclude them.

### 13. Is the dataset balanced?

No. We have 4,550 Standard and 700 Abrupt windows. We account for the imbalance with class-weighted training and report macro F1 and balanced accuracy in addition to accuracy.

## Preprocessing and model

### 14. What is a window?

A fixed duration of the continuous sensor sequence. Here each window covers 3 seconds. At 50 samples per second, it contains 150 readings per channel.

### 15. Why use a CNN-LSTM?

The 1D CNN detects short/local patterns such as a sudden acceleration or rotation change. The LSTM processes how those patterns unfold over time. Combining them suits sequential sensor signals.

### 16. What does feature fusion mean in this project?

The six acceleration and gyroscope axes are input together. The network can learn combinations across the channels. We do not fuse all five body sensors; that is beyond this first model.

### 17. What is the difference between a 1D CNN and a 2D image CNN?

A 1D convolution slides along time and learns temporal patterns in sensor channels. A 2D image CNN slides across image height and width. Our input is a time series, not an image.

### 18. What is the role of the LSTM?

It maintains a learned representation of earlier sequence information while reading later time steps, so the model can use temporal order rather than treating each reading independently.

### 19. Why a bidirectional LSTM?

Each fixed window is classified after all 3 seconds are collected. A bidirectional LSTM reads the window forward and backward, giving each time point context from both directions. It is suitable for window classification but is not a strictly causal streaming LSTM within the window.

### 20. What is normalization and why do it?

Sensor axes have different ranges and offsets. We scale each channel using mean and standard deviation learned from the fitting participants. Validation/test and phone data use those same training values so evaluation does not leak their statistics into training.

### 21. How do you handle noise and imbalance?

Training uses class-weighted cross-entropy to give the rarer Abrupt class more influence, plus small jitter and gain variation to model sensor noise/scale differences. We do not claim these steps remove every source of noise.

### 22. What is data augmentation here?

Small training-only perturbations to sensor values (jitter and gain changes). They create plausible variations of existing training windows; they do not add new participants or replace real data.

## Evaluation and results

### 23. How did you prevent data leakage?

We split by participant. A participant's windows are assigned wholly to fitting, validation, or test. This is more realistic than randomly splitting overlapping windows from the same person across groups.

### 24. How many participants are in each split?

36 fitting, 12 validation, and 12 held-out test participants (60%, 20%, 20%).

### 25. What are the final test results?

On 1,080 windows from held-out participants: accuracy 97.1%, macro F1 93.7%, balanced accuracy 93.3%. Standard F1 was 98.3%; Abrupt F1 was 89.1%, with 88.2% Abrupt recall.

### 26. Why report macro F1 if accuracy is high?

Accuracy can look strong when the common class is predicted well. Macro F1 averages class F1 scores equally, making it more informative when one class has many fewer examples.

### 27. Define precision and recall.

Precision asks: of the windows predicted as a class, what fraction were correct? Recall asks: of the real windows in that class, what fraction did the model find?

### 28. What does Abrupt recall of 88.2% mean?

On the held-out DASIG test windows labeled Abrupt, the model correctly identified about 88.2%. It missed the remainder. This is not a guarantee about alarms or real factory incidents.

### 29. Why is the split by person important?

Adjacent sensor windows from one person can be very similar. If some went into training and others into test, the score could reward memorizing person-specific patterns. Holding participants out tests transfer to unseen people more directly.

### 30. How fast is inference?

The model-only CPU timing was about 0.62 ms per 3-second window on our development computer. It excludes browser sensor sampling, Wi-Fi/network transfer, and display update; do not call it total end-to-end latency.

## Phone demonstration

### 31. What does the phone part do?

The phone browser reads accelerometer/gyroscope events, resamples them into rolling 50 Hz windows, sends a window to the laptop server, and displays the model's class scores. If the user selects an expected class and records, the phone sends labeled samples to the laptop CSV folder for a separate dashboard check.

### 32. Is the phone used to train the model?

No. It runs the already-trained DASIG model. Phone recordings are saved and scored separately; they are not added to training automatically.

### 33. Why can the phone prediction differ from the DASIG benchmark?

DASIG sensors are worn on the forearm; the phone may be held in a hand or strapped differently. Axes, sensor calibration, gravity handling, sampling frequency, phone model, and user motion all differ. This is a domain shift.

### 34. Can we report the phone's accuracy?

Yes, as a separate labeled phone-session check for the people/device used. Do not combine it with the participant-held-out DASIG score or claim it generalizes to other phones/users.

### 35. Why choose a label before recording?

The expected label provides a reference for checking the model's prediction. The model itself predicts from motion readings; it does not see the label control as an input. Unknown periods and windows spanning label changes are skipped in the metric calculation.

### 36. Why must I record each action for at least 3 seconds?

The model requires a complete 3-second input window. Longer stable recordings produce multiple overlapping windows for checking.

### 37. Why does Android Chrome need a local flag?

Browsers restrict motion-sensor access on insecure web pages. The current laptop server uses local HTTP; the Android Chrome flag treats that exact local origin as secure for development. Use it only for this local demonstration and disable it afterward. This setup is not public production hosting.

### 38. Does the phone predict entirely on-device?

No. The web page captures phone sensors, but the laptop Python server runs the neural network. Both devices must be reachable on the same Wi-Fi during the live demo.

## Questions from teammates / leadership

### 39. What should each group member present?

One person can introduce the use case/scope, one explain dataset labels and preprocessing, one cover CNN-LSTM and evaluation, and one run the phone demo. Everyone should know the two class definitions and the proxy-label limitation.

### 40. What is the most important caveat to say unprompted?

The accuracy is on DASIG's event-derived two-class task. It is not factory deployment accuracy, and the phone check is separate because the hardware and sensor placement differ.

### 41. What are the main strengths?

The task is clearly scoped; the input is multi-axis time-series data; the CNN-LSTM models both local and sequential patterns; the split holds out participants; and the report includes class-sensitive metrics.

### 42. What are the main limitations?

Only two classes; event-derived approximate labels; simulated workstation; one forearm sensor rather than all five; limited representation of users/tasks; and phone domain shift. No production validation or uncertainty calibration is claimed.

### 43. What would you improve next?

Collect accurately labeled data from more users and real equipment; define clear gesture classes and onset/offset rules; test additional body positions; compare CNN-only, LSTM-only, and CNN-LSTM baselines; tune thresholds using validation data; and evaluate latency and false alarms in the deployment setup.

### 44. How could you reduce false alarms?

Use validation data to select a class threshold, require repeated/high-confidence detections, and measure false alarms per unit time on realistic continuous recordings. Our current window benchmark does not fully measure an operational false-alarm rate.

### 45. Is there an alarm system in our product?

No. DASIG alarm cues are part of the experiment and help derive labels. The model output is a movement class; we have not built an industrial alert/actuation system.

### 46. What should we say if asked why Abrupt recall is lower?

Abrupt is the minority class and its examples vary. Event-derived labels can include reaction delay or mixed motion. Class weighting helps but does not eliminate these issues. We should report the per-class result honestly.

### 47. What does the confidence score mean?

It is the softmax score for a class from this model. It is not guaranteed to be a calibrated probability of correctness. Calibration would need a separate analysis.

### 48. Does a high score prove the model understands a gesture?

No. It shows predictions align with the available labels on the held-out DASIG participants. Data quality, label definitions, confounding factors, and external validation still matter.

### 49. Can it identify a person?

That is not the task and we do not train an identity classifier. Sensor patterns can still vary by person, which is why participant-held-out evaluation matters.

### 50. Why not use a traditional machine-learning model?

The assigned project calls for a neural sequence model. A traditional feature-based baseline would be useful for comparison and could be simpler, but it is not the model currently reported.

### 51. Is the model real time?

The live phone page produces predictions repeatedly from rolling windows, and model-only inference is fast on the laptop. It needs 3 seconds to fill the first window and depends on phone/browser/network timing, so we describe it as a real-time-style demonstration, not a validated industrial real-time system.

### 52. What exactly did the group build versus reuse?

We use the public DASIG dataset and PyTorch/Streamlit libraries. We implemented the dataset parsing/window-labeling pipeline, CNN-LSTM training/evaluation, local prediction server, phone motion page, dashboard, and project documentation.

### 53. Where are the exact numbers and code explanations?

The metrics are in `report/experiment_results.md`. The leaders guide is in `docs/LEADERS_GUIDE.md`, and file responsibilities are in `docs/CODE_EXPLANATION.md`.

## Demo failure fallback

If phone motion access or Wi-Fi fails, present the DASIG benchmark tab and its held-out example prediction. Explain that the phone page requires browser sensor permission and local connectivity; the dataset/model evaluation does not depend on a phone.
