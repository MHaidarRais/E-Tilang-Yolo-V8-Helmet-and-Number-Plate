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
        filename = f"{date_str}.jpg"
        try:
            image_url = storage.child(f"ViolationCaptured/{filename}").get_url(None)
            violation['image'] = image_url
        except:
            violation['image'] = None
    
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
    filename = f"{date_str}.jpg"
    try:
        image_url = storage.child(f"ViolationCaptured/{filename}").get_url(None)
        violation['image'] = image_url
    except:
        violation['image'] = None
    
    return render_template('violation_details.html', violation=violation)
