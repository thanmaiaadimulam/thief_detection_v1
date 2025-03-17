# Zone Security System

A computer vision-based security monitoring system that allows users to define protected zones and detect intrusions using OpenCV, hand tracking, and pose detection.

## Features

- **Password Protection**: Secure system access with password authentication
- **Flexible Zone Setting**: Define protected areas using either:
  - Hand gesture pinch detection
  - Mouse click-and-drag interface
- **Real-time Intrusion Detection**: Monitor for unauthorized entry into protected zones
- **Automated Evidence Capture**:
  - Records video when intrusions are detected
  - Captures still images for documentation
  - Timestamps all evidence files
- **Visual Feedback**: On-screen indicators for system state and alerts

## System Flow

1. **System Activation**: Press 'b' to activate the system
2. **Authentication**: Enter password to gain access
3. **Zone Setting**: Define the protected area through gestures or mouse
4. **Monitoring**: System actively monitors for intrusions
5. **Alert & Recording**: Automatic evidence capture when intrusions detected
6. **Zone Adjustment**: Option to modify zone boundaries as needed

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- Custom modules:
  - `tracking_module.py` (Hand tracking functionality)
  - `pose_module.py` (Body pose detection)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/thanmaiaadimulam/thief_detection_v1.git
   cd zone-security-system
   ```

2. Install dependencies:
   ```
   pip install opencv-python numpy
   ```

3. Ensure the required tracking modules are in the project directory.

## Usage

Run the main application:
```
python zonesetting.py
```

### Key Controls:
- 'b': Activate system / Enter verification mode
- 's': Save zone and start monitoring
- 'o': Enter zone offset/adjustment mode 
- 'q': Quit the application

### Setting a Zone:
1. Press 'b' and enter the default password (`admin123`)
2. Use one of the following methods:
   - **Mouse**: Click and drag to draw a rectangle
   - **Hand Gesture**: Make a pinch gesture with all fingers up to set points

3. Press 's' to save the zone and begin monitoring

### Security Notes:
- Default password is `admin123` - change this in production use
- Evidence is saved in the `security_captures` directory
- Video recordings stop automatically after 10 seconds of detected intrusion

## Customization

You can modify the following aspects:
- Password: Change the default password in the `initialize_password` method
- Recording Duration: Adjust the recording time in the monitoring section
- Zone Visualization: Modify colors and line thickness for different visual feedback

## Project Structure

- `zonesetting.py`: Main application file
- `tracking_module.py`: Hand tracking and gesture detection
- `pose_module.py`: Body pose detection
- `security_captures/`: Directory for saved evidence

## Troubleshooting

**Camera Issues**:
If the camera fails to initialize, ensure no other applications are using it and check device permissions.

**OpenCV Errors**:
Some MacOS users might encounter Tkinter-related errors. The current implementation uses an OpenCV-based dialog to avoid these issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with OpenCV for computer vision processing
- Uses hand and pose detection modules for advanced tracking
