from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pandas as pd
from rapidfuzz import process, fuzz
import logging
import cv2
import time
import os
import numpy as np
from advance_face_recognition import verify_user_image,is_name_not_in_list
import mysql.connector
from datetime import datetime
import logs_db
import images_space
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variable to store current username
current_username = None
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
ROLE_MAP = {
    0: "x",
    1: "y",
    2: "z"
}

def local_screening(name, filepath, threshold=60):
    logging.info(f"local_screening called with name='{name}', file='{filepath}'")
    try:
        df = pd.read_excel(filepath, usecols=[0], header=0)
        names_list = df.iloc[:, 0].dropna().astype(str).tolist()
        logging.debug(f"Loaded {len(names_list)} names from {filepath}")

        best_match = process.extractOne(name, names_list, scorer=fuzz.ratio)
        logging.debug(f"Best match result: {best_match}")

        if best_match and best_match[1] >= threshold:
            logging.info(
                f"Name '{name}' matched '{best_match[0]}' with score {best_match[1]}"
            )
            return {
                "status": "success",
                "match_name": best_match[0],
                "score": best_match[1],
            }

        logging.warning(f"No match or below threshold for '{name}'")
        return {
            "status": "failed",
            "match_name": best_match[0] if best_match else None,
            "score": best_match[1] if best_match else 0,
            "message": "Identity verification failed. Name not recognized or below threshold.",
        }

    except Exception as e:
        logging.error(f"Error in local_screening: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing request: {str(e)}",
        }


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def checkface(image_path):
    global current_username
    logging.info(f"checkface called with image_path: {image_path}, name: {current_username}")

    if not current_username:
        return {"verified": False, "status": "denied", "message": "Username missing"}

    if not os.path.exists(image_path):
        return {"verified": False, "status": "denied", "message": "Image file not found"}
    if not is_name_not_in_list(current_username):
        return {"verified": False, "status": "denied", "message": "access denied"}

    try:
        result = verify_user_image(
            claimed_user=current_username,
            image_path=image_path,
            out_dir="user_templates",
            model="hog",
            threshold=0.91
        )
        decision = result.get("decision", False)
        acc = result.get("score", 0.99)

        rid = logs_db.log_face_event(current_username, "203.0.113.42", decision, acc)
        images_space.upload_to_spaces(image_path, key=f"{rid}.jpg")

        if decision:
            logging.info(f"Face verification successful for {current_username}, image: {image_path}")
            logs_db.update_last_user(current_username)
            return {"verified": True, "status": "approved", "name": current_username}
        else:
            logging.info(f"Face verification failed for {current_username}, image: {image_path}")
            return {"verified": False, "status": "denied", "name": current_username}

    except Exception as e:
        logging.error(f"Error during face verification for {current_username}: {e}")
        return {"verified": False, "status": "denied", "message": str(e)}


def capture_good_face(save_path="static/captured_face.jpg", timeout=15):
    """Enhanced face capture function with better camera handling and face detection"""
    logging.info(f"Starting face capture, save_path: {save_path}")

    # Ensure static directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cap = None

    # Try multiple camera backends for better compatibility
    backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        logging.info(f"Trying camera backend: {backend}")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                logging.info(f"Successfully opened camera with backend: {backend}")
                break
            cap.release()
            cap = None
        else:
            if cap:
                cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        logging.error("Could not open camera with any backend")
        return None

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_time = time.time()
    frame_count = 0
    best_face_data = None
    best_quality_score = 0

    logging.info("Starting face detection loop...")
    time.sleep(0.5)  # warm up

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to read frame {frame_count + 1}")
                time.sleep(0.1)
                continue

            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            logging.info(f"Frame {frame_count}: Found {len(faces)} faces")

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    area = w * h
                    center_x = x + w // 2
                    center_y = y + h // 2
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2

                    center_distance = (
                                              (center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2
                                      ) ** 0.5

                    quality_score = area / (1 + center_distance * 0.01)
                    logging.info(
                        f"Face quality score: {quality_score}, area: {area}"
                    )

                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        padding = 20
                        y1 = max(0, y - padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        x1 = max(0, x - padding)
                        x2 = min(frame.shape[1], x + w + padding)

                        best_face_data = frame[y1:y2, x1:x2]
                        logging.info(
                            f"New best face found with score: {quality_score}"
                        )

                    if quality_score > 20000:  # Threshold for good quality
                        cv2.imwrite(save_path, best_face_data)
                        if os.path.exists(save_path):
                            file_size = os.path.getsize(save_path)
                            logging.info(
                                f"Face capture successful. File saved: {save_path}, Size: {file_size} bytes"
                            )
                            cap.release()
                            return save_path
                        else:
                            logging.error(
                                f"Failed to save file at {save_path}"
                            )

            if time.time() - start_time > timeout:
                logging.warning("Face capture timed out")
                break

            time.sleep(0.1)

    except Exception as e:
        logging.error(f"Error during face capture: {str(e)}", exc_info=True)
    finally:
        if cap:
            cap.release()

    if best_face_data is not None:
        try:
            cv2.imwrite(save_path, best_face_data)
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                logging.info(
                    f"Best available face saved: {save_path}, Size: {file_size} bytes"
                )
                return save_path
        except Exception as e:
            logging.error(f"Error saving best face: {str(e)}")

    logging.error("No suitable face found within timeout period")
    return None


@app.route('/')
def index():
    logging.info("GET / called")
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    logging.info("GET /dashboard called")
    return render_template('dashboard.html', user_name=current_username)


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the current session and clear stored username"""
    global current_username
    logging.info("POST /reset_session called")

    old_username = current_username
    current_username = None

    logging.info(f"Session reset: cleared username '{old_username}'")
    return jsonify({
        "status": "success",
        "message": "Session reset successfully",
        "previous_username": old_username
    })


@app.route('/check_name', methods=['POST'])
def check_name():
    global current_username
    logging.info("POST /check_name called")
    username = request.form.get('username', '').strip()
    logging.debug(f"Username received: '{username}'")

    filepath = "data.xlsx"
    result = local_screening(username, filepath, threshold=60)

    if result["status"] == "success":
        current_username = result["match_name"]  # Save verified username
        logging.info(f"Stored username: {current_username}")

    return jsonify(result)


@app.route('/auth', methods=['POST'])
def auth():
    global current_username
    logging.info("POST /auth called")
    username = request.form.get('username', '').strip()
    logging.debug(f"Username received: '{username}'")

    filepath = "data.xlsx"
    result = local_screening(username, filepath, threshold=60)

    if result["status"] == "success":
        current_username = result["match_name"]  # Store the verified username
        logging.info(f"Authentication successful for '{result['match_name']}'")
        return jsonify({
            "status": "success",
            "match_name": result["match_name"],
            "score": result["score"],
        })
    else:
        # Clear username on authentication failure
        current_username = None
        logging.warning("Authentication failed - cleared username")
        return jsonify({
            "status": "failed",
            "message": result.get("message", "Identity verification failed."),
            "score": result.get("score", 0),
        })


@app.route("/upload_picture", methods=["POST"])
def upload_picture():
    """Upload a picture and send it to the face verification pipeline"""
    global current_username
    logging.info(f"POST /upload_picture called with current_username: {current_username}")

    try:
        uploaded_file = request.files.get("picture")
        if not uploaded_file:
            logging.error("No picture file in request")
            return jsonify({"success": False, "message": "No picture uploaded"}), 400

        if not current_username:
            logging.error("No current username set for upload verification")
            return jsonify({"success": False, "message": "No user session found"}), 400

        # Save uploaded file temporarily
        save_path = "static/uploaded_face.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        uploaded_file.save(save_path)

        # Directly run face verification
        verification_result = checkface(save_path)

        size = os.path.getsize(save_path)
        logging.info(
            f"Picture saved at {save_path} ({size} bytes), verification: {verification_result}"
        )

        return jsonify({
            "success": True,
            "path": f"/{save_path}",
            "image_path": save_path,
            "file_size": size,
            "verification_result": verification_result,
        })

    except Exception as e:
        logging.error(f"Error in upload_picture route: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Server error during picture upload: {str(e)}",
        }), 500


@app.route("/capture_face", methods=["POST"])
def capture_face():
    global current_username
    logging.info(f"POST /capture_face called with current_username: {current_username}")

    try:
        if not current_username:
            logging.error("No current username set for face capture verification")
            return jsonify({"success": False, "message": "No user session found"}), 400

        # 1) read uploaded image from the browser
        uploaded = request.files.get("image")
        if not uploaded:
            logging.error("No image file in request")
            return jsonify({"success": False, "message": "No image uploaded"}), 400

        image_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("Failed to decode uploaded image")
            return jsonify({"success": False, "message": "Invalid image"}), 400

        # 2) detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        logging.info(f"Uploaded image: found {len(faces)} faces")
        if len(faces) == 0:
            return jsonify({"success": False, "message": "No face detected"}), 200

        # 3) save and return
        save_path = "static/captured_face.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

        verification_result = checkface(save_path)

        size = os.path.getsize(save_path)
        logging.info(
            f"Face saved: {save_path} ({size} bytes), verification: {verification_result}"
        )
        return jsonify({
            "success": True,
            "path": f"/{save_path}",
            "image_path": save_path,
            "file_size": size,
            "verification_result": verification_result,
        })

    except Exception as e:
        logging.error(f"Error in capture_face route: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Server error during face capture: {str(e)}",
        }), 500

# DB connection (replace with actual credentials)
def get_connection():
    return mysql.connector.connect(
        host="dewan-db-mysql-do-user-19563317-0.e.db.ondigitalocean.com",
        port="25060",
        user="doadmin",
        password="AVNS_GFSrKBJqdaBaBTZ056V",
        database="dewan_db",
        ssl_disabled=True
    )

@app.route("/get_tasks", methods=["GET"])
def get_tasks():
    conn = get_connection()
    tasks = []
    with conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, task_description, frequency, next_run, role_text, task_title FROM scheduled_tasks")
        tasks = cursor.fetchall()
        for task in tasks:
            task["role_text"] = ROLE_MAP.get(int(task["role_text"]), "unknown")
        cursor.close()
    return jsonify(tasks)

@app.route("/get_emails", methods=["GET"])
def get_emails():
    conn = get_connection()
    emails = []
    with conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, email, role_text, errors_count FROM tasks_emails")
        emails = cursor.fetchall()
        for email in emails:
            email["role_text"] = ROLE_MAP.get(int(email["role_text"]), "unknown")
        cursor.close()
    return jsonify(emails)
@app.route("/add_task", methods=["POST"])
def add_task():
    data = request.json
    required_fields = ["task_description", "frequency", "next_run", "role_text", "task_title"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO scheduled_tasks (task_description, frequency, next_run, role_text, task_title) VALUES (%s, %s, %s, %s, %s)",
            (
                data["task_description"],
                data["frequency"],
                data["next_run"],
                data["role_text"],
                data["task_title"]
            ))
            conn.commit()
    return jsonify({"message": "Task added successfully"}), 201


@app.route("/add_email", methods=["POST"])
def add_email():
    data = request.json
    required_fields = ["email", "role_text"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO tasks_emails (email, role_text, errors_count) VALUES (%s, %s, 0)",
            (
                data["email"],
                data["role_text"]
            ))
            conn.commit()
    return jsonify({"message": "Email added successfully"}), 201

@app.route("/remove_task", methods=["POST"])
def remove_task():
    data = request.json
    task_id = data.get("id")
    if not task_id:
        return jsonify({"error": "Missing 'id' field"}), 400

    conn = get_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM scheduled_tasks WHERE id = %s", (task_id,))
            conn.commit()
    return jsonify({"message": f"Task with ID {task_id} removed successfully."}), 200
@app.route('/automation')
def automation():
    username = request.args.get('username', 'Guest')  # gets ?username=... from URL
    return render_template('automation.html', user_name=username)
@app.route("/remove_email", methods=["POST"])
def remove_email():
    data = request.json
    email_id = data.get("id")
    if not email_id:
        return jsonify({"error": "Missing 'id' field"}), 400

    conn = get_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM tasks_emails WHERE id = %s", (email_id,))
            conn.commit()
    return jsonify({"message": f"Email with ID {email_id} removed successfully."}), 200


#new
def fetch_face_logs():
    try:
        return logs_db.fetch_face_logs()
    except Exception as e:
        print(e)
        return [
            {'id': 7, 'user_name': 'alice', 'ip': '203.0.113.42', 'status': 0, 'accuracy': 0.99,
             'date_time': datetime(2025, 9, 15, 17, 10, 49)},
            {'id': 8, 'user_name': 'bob', 'ip': '192.168.1.100', 'status': 1, 'accuracy': 0.95,
             'date_time': datetime(2025, 9, 15, 16, 30, 22)},
            {'id': 9, 'user_name': 'charlie', 'ip': '10.0.0.50', 'status': 0, 'accuracy': 0.87,
             'date_time': datetime(2025, 9, 15, 15, 45, 10)},
        ]


def fetch_users():
    try:
        return logs_db.fetch_users()
    except Exception:
        return [
            {'name': 'Adriana Lima'},
            {'name': 'Alex Lawther'},
            {'name': 'Alexandra Daddario'}
        ]


def delete_user(name: str):
    try:
        logs_db.delete_user(name)
        return True
    except Exception:
        print(f"Deleting user: {name}")
        return True


def enroll_user_from_folder(user, folder_path, out_dir="user_templates", k_max=5, min_per_cluster=8, overwrite=False,
                            verbose=True):
    return {
        'user': user,
        'status': 'created',
        'templates_path': f'user_templates\\{user}.npy',
        'k': 2,
        'n_images': 10,
        'n_usable': 10
    }



@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == '123456':
            session['logged_in'] = True
            return redirect(url_for('dashboard_admin'))
        else:
            flash('Invalid credentials')

    return render_template('login_admin.html')


@app.route('/dashboard_admin')
def dashboard_admin():
    if 'logged_in' not in session:
        return redirect(url_for('login_admin'))

    logs = fetch_face_logs()
    return render_template('dashboard_admin.html', logs=logs)


@app.route('/users_admin')
def users_admin():
    if 'logged_in' not in session:
        return redirect(url_for('login_admin'))

    users_list = fetch_users()
    return render_template('users_admin.html', users=users_list)


@app.route('/delete_user_admin/<name>', methods=['POST'])
def delete_user_admin(name):
    if 'logged_in' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})

    try:
        delete_user(name)
        return jsonify({'success': True, 'message': f'User {name} deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/add_user_admin', methods=['POST'])
def add_user_admin():
    if 'logged_in' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})

    try:
        user = request.form['user']
        folder_path = request.form['folder_path']
        overwrite = 'overwrite' in request.form

        result = enroll_user_from_folder(
            user=user,
            folder_path=folder_path,
            overwrite=overwrite
        )

        return jsonify({'success': True, 'message': f'User {user} added/updated successfully', 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/logout_admin')
def logout_admin():
    session.pop('logged_in', None)
    return redirect(url_for('login_admin'))


@app.route("/face-recognition", methods=["POST"])
def verify():
    claimed_user = request.form.get("name")
    image = request.files.get("image")

    if not claimed_user or not image:
        return jsonify({"error": "Missing name or image"}), 400

    # Save uploaded image securely
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)

    # Call your verification function
    res = verify_user_image(
        claimed_user=claimed_user,
        image_path=image_path,
        out_dir="user_templates",
        model="hog",
        threshold=0.91
    )

    return jsonify(res)

if __name__ == '__main__':
    logging.info("Starting Flask app on port 5000")
    app.run(debug=True, port=8000)