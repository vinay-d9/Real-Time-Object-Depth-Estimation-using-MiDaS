# Viva Questions and Answers

1. **What is depth estimation?** It predicts how far image regions are from the camera.
2. **What is monocular depth estimation?** It estimates depth from one RGB image instead of stereo cameras or a depth sensor.
3. **What is MiDaS?** MiDaS is a pre-trained model for monocular relative-depth estimation.
4. **Why use MiDaS?** It is open-source, already trained, works on ordinary RGB images, and is designed for cross-scene relative depth.
5. **Was MiDaS trained in this project?** No. The project only loads published pre-trained weights and performs inference.
6. **What does pre-trained mean?** The model learned its parameters earlier from training data and can be reused for prediction.
7. **What is inference?** Running an input through the trained model to obtain a prediction.
8. **What is transfer learning?** Reusing learned model knowledge for another task. This project uses a pre-trained model directly rather than fine-tuning it.
9. **What is the input?** One RGB image represented as a transformed PyTorch tensor with a batch dimension.
10. **What is the output?** A dense relative-depth or inverse-depth-like prediction for image locations.
11. **What is relative depth?** It indicates ordering such as nearer versus farther, without a guaranteed physical scale.
12. **Why not say an object is 5 meters away?** The model has no scene-scale calibration or direct distance sensor.
13. **Why convert BGR to RGB?** OpenCV reads color channels as BGR, while the model transform expects RGB.
14. **Why normalize the input?** To match the value distribution used when the pre-trained model was built.
15. **Why resize the output?** The model processes a transformed resolution, so resizing aligns the prediction with the original image.
16. **Why use `model.eval()`?** It switches the network to prediction behavior and disables training-specific updates.
17. **Why use inference mode?** Gradients are unnecessary for prediction, so inference mode saves memory and computation.
18. **What is an encoder?** The feature-extraction part that converts pixels into useful visual representations.
19. **What is a decoder?** The part that combines features and reconstructs a dense spatial prediction.
20. **What is a depth map?** An image whose pixels represent estimated depth-related values.
21. **How is it visualized?** Values are normalized and mapped to colors with the Inferno colormap.
22. **What affects speed?** Model size, input resolution, device, memory, and video decoding overhead.
23. **Can it work in real time?** It can process streams only as fast as the hardware permits; the project reports actual timing instead of promising a fixed FPS.
24. **What are limitations?** Relative scale ambiguity, lighting and occlusion errors, difficult transparent or reflective objects, and CPU latency.
25. **How could actual distance be estimated?** Use a calibrated metric-depth model, stereo geometry, camera calibration, or a depth sensor.
26. **How would you improve the project?** Add metric calibration, faster inference, object detection, sensor fusion, and edge deployment.