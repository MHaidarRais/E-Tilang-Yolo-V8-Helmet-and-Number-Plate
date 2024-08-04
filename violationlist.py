# violationslist.py

from flask import Blueprint, render_template
from firebaselib import db, storage

violations_bp = Blueprint('violations', __name__)

@violations_bp.route('/violations')
def violations():
    violations_ref = db.collection('ETLE')
    violations = [violation.to_dict() for violation in violations_ref.stream()]

    for violation in violations:
        day = int(violation['day'])
        month = int(violation['month'])
        year = int(violation['year'])
        hour = int(violation['hour'])
        minute = int(violation['minute'])
        second = int(violation['second'])
        date_str = f"{day:02d}-{month:02d}-{year:04d} {hour:02d}-{minute:02d}-{second:02d}"
        violation_filename = f"{date_str}.jpg"
        numberplate_filename = f"numberplate_{date_str}.jpg"
        try:
            image_url = storage.child(f"ViolationCaptured/{violation_filename}").get_url(None)
            violation['image'] = image_url
        except:
            violation['image'] = None
        try:
            numberplate_image_url = storage.child(f"NumberPlateCaptured/{numberplate_filename}").get_url(None)
            print(f"Number Plate Image URL for {numberplate_filename}: {numberplate_image_url}")  # Debugging print
            violation['number_plate_image'] = numberplate_image_url
        except Exception as e:
            print(f"Error fetching number plate image for {numberplate_filename}: {e}")  # Debugging print
            violation['number_plate_image'] = None
        
        # Add detected number plate information
        violation['number_plate'] = violation.get('number_plate', 'N/A')  # Assuming 'number_plate' field exists
    
    return render_template('violationList.html', violations=violations)

@violations_bp.route('/violation/<int:index>')
def violation_details(index):
    violations_ref = db.collection('ETLE')
    violations = [violation.to_dict() for violation in violations_ref.stream()]
    
    if index < 0 or index >= len(violations):
        return "Invalid violation index."
    
    violation = violations[index]
    day = int(violation['day'])
    month = int(violation['month'])
    year = int(violation['year'])
    hour = int(violation['hour'])
    minute = int(violation['minute'])
    second = int(violation['second'])
    date_str = f"{day:02d}-{month:02d}-{year:04d} {hour:02d}-{minute:02d}-{second:02d}"
    violation['number_plate'] = violation.get('number_plate', 'N/A')
    violation_filename = f"{date_str}.jpg"
    numberplate_filename = f"numberplate_{date_str}.jpg"
    try:
        image_url = storage.child(f"ViolationCaptured/{violation_filename}").get_url(None)
        violation['image'] = image_url
    except:
        violation['image'] = None
    
    try:
            numberplate_image_url = storage.child(f"NumberPlateCaptured/{numberplate_filename}").get_url(None)
            print(f"Number Plate Image URL for {numberplate_filename}: {numberplate_image_url}")  # Debugging print
            violation['number_plate_image'] = numberplate_image_url
    except Exception as e:
            print(f"Error fetching number plate image for {numberplate_filename}: {e}")  # Debugging print
            violation['number_plate_image'] = None
    
    
    return render_template('violation_details.html', violation=violation)
