import os
import sys
from codecarbon import EmissionsTracker

# Ensure the backend directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import inference

def profile_with_codecarbon():
    print("Initializing CodeCarbon Emissions Tracker...")
    
    # Ensure the metrics storage directory exists before initializing the tracker
    metrics_dir = "tests/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Configure the hardware tracker
    tracker = EmissionsTracker(
        project_name="pulmonet_inference_energy",
        measure_power_secs=1,
        save_to_file=True,
        output_dir=metrics_dir,
        log_level="warning" # Suppress verbose tracking print statements
    )
    
    # Dummy mock data for a standard application test run
    image_path = "backend/storage/patient_images/test_sample.jpg" 
    notes = "Patient presents with persistent productive cough and fever."
    wbc = 14.5
    crp = 85.0

    print("Starting energy monitoring execution...")
    tracker.start()
    
    try:
        # Run your multimodal joint sequence pool forward pass
        result = inference.predict(image_path, notes, wbc, crp)
        print(f"Inference Complete. Prediction: {result['label']} ({result['probability']:.2f})")
    finally:
        # Guarantee tracker termination even if inference crashes
        emissions_data = tracker.stop()
        print(f"Tracking stopped. Estimated Net Emissions: {emissions_data} kg CO2")
        print("Detailed metrics saved to tests/metrics/emissions.csv")

if __name__ == "__main__":
    profile_with_codecarbon()