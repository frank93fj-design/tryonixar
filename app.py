import os
import json
import base64
import requests
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai.types import RawReferenceImage, EditImageConfig, Image
import tempfile
import time as _time

app = Flask(__name__)

# ── CLOUD CONFIG ──
# Updated to match your JSON key exactly
PROJECT_ID = "tryonixar-494001"
LOCATION = "us-central1"

# ── GOOGLE CREDENTIALS HANDLING ──
_gcp_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if _gcp_json:
    try:
        # Repairing newline escaping for Railway environment variables
        formatted_json = _gcp_json.replace('\\n', '\n')
        
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _tmp.write(formatted_json)
        _tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name
        print(f"Vertex AI credentials file created successfully.")
    except Exception as e:
        print(f"FAILED to create credentials file: {e}")
else:
    print("WARNING: GOOGLE_APPLICATION_CREDENTIALS_JSON is missing!")

# Initialize Vertex Client
try:
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    print(f"Vertex AI initialized: {PROJECT_ID}")
except Exception as e:
    print(f"Vertex AI Client failed to initialize: {e}")

# ── MESHY CONFIG ──
MESHY_API_KEY = os.environ.get("MESHY_API_KEY")
MESHY_BASE_URL = "https://api.meshy.ai/openapi/v1"

# ── FOLDERS ──
UPLOAD_FOLDER = 'static/uploads'
MODELS_FOLDER = 'static/uploads/models'
WARDROBE_FOLDER = 'static/wardrobe'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(WARDROBE_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

def get_wardrobe_items():
    return [
        f for f in os.listdir(WARDROBE_FOLDER)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]

def filename_to_garment_description(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    tokens = stem.replace('_', '-').split('-')
    sleeve_map = {
        ('long', 'sleeve'):  'with long sleeves',
        ('short', 'sleeve'): 'with short sleeves',
        ('half', 'sleeve'):  'with half sleeves',
        ('3quarter', 'sleeve'): 'with three-quarter sleeves',
        ('sleeveless',):     'sleeveless',
        ('cropped',):        'cropped',
        ('oversized',):      'oversized',
    }
    remaining = tokens[:]
    sleeve_clause = ''
    for key_tuple, clause in sleeve_map.items():
        key_list = list(key_tuple)
        n = len(key_list)
        for i in range(len(remaining) - n + 1):
            if remaining[i:i + n] == key_list:
                sleeve_clause = clause
                remaining = remaining[:i] + remaining[i + n:]
                break
        if sleeve_clause: break

    base_description = ' '.join(remaining)
    if not sleeve_clause: return base_description
    if sleeve_clause in ('sleeveless', 'cropped', 'oversized'):
        return f"{sleeve_clause} {base_description}"
    return f"{base_description} {sleeve_clause}"

# ── ROUTES ──

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wardrobe')
def wardrobe():
    return render_template('Wardrobe.html', wardrobe_items=get_wardrobe_items())

@app.route('/generate_3d', methods=['POST'])
def generate_3d():
    if not MESHY_API_KEY:
        return jsonify({'error': 'Meshy API key missing'}), 500
    photo = request.files.get('photo')
    if not photo:
        return jsonify({'error': 'No photo provided'}), 400
    user_img_path = os.path.join(UPLOAD_FOLDER, 'user_base.jpg')
    photo.save(user_img_path)
    headers = {'Authorization': f'Bearer {MESHY_API_KEY}'}
    try:
        with open(user_img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        image_data_uri = f"data:image/jpeg;base64,{b64}"
        job_res = requests.post(
            f'{MESHY_BASE_URL}/image-to-3d',
            headers={**headers, 'Content-Type': 'application/json'},
            json={'image_url': image_data_uri, 'enable_pbr': True},
            timeout=30,
        )
        job_res.raise_for_status()
        return jsonify({'job_id': job_res.json().get('result') or job_res.json().get('id')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meshy_status/<job_id>', methods=['GET'])
def meshy_status(job_id):
    headers = {'Authorization': f'Bearer {MESHY_API_KEY}'}
    try:
        res = requests.get(f'{MESHY_BASE_URL}/image-to-3d/{job_id}', headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()
        status = data.get('status', 'PENDING')
        model_url = None
        if status == 'SUCCEEDED':
            glb_url = data.get('model_urls', {}).get('glb')
            if glb_url:
                glb_path = os.path.join(MODELS_FOLDER, f'{job_id}.glb')
                glb_res = requests.get(glb_url, timeout=60)
                with open(glb_path, 'wb') as f: f.write(glb_res.content)
                model_url = f'/static/uploads/models/{job_id}.glb'
        return jsonify({'status': status, 'progress': data.get('progress', 0), 'model_url': model_url})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'UNKNOWN'}), 500

@app.route('/upload_glb', methods=['POST'])
def upload_glb():
    glb_file = request.files.get('glbFile')
    if not glb_file: return jsonify({'success': False}), 400
    safe_name = os.path.basename(glb_file.filename)
    glb_path = os.path.join(MODELS_FOLDER, safe_name)
    glb_file.save(glb_path)
    return jsonify({'success': True, 'url': f'/static/uploads/models/{safe_name}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('imageUpload')
    selected_items_str = request.form.get('selectedItems', '[]')
    if not file: return "No snapshot provided", 400

    user_img_path = os.path.join(UPLOAD_FOLDER, 'user_base.jpg')
    file.save(user_img_path)

    selected_list = json.loads(selected_items_str)
    if not selected_list: return "No clothes selected", 400

    garment_filename = selected_list[0]
    garment_description = filename_to_garment_description(garment_filename)

    try:
        person_img = Image.from_file(location=user_img_path)
        person_ref = RawReferenceImage(reference_image=person_img, reference_id=0)

        # RESTORED: Your original deep prompt structure
        prompt = (
            "TASK: Photorealistic virtual clothing try-on.\n"
            "A reference photo of a specific real person is provided. "
            "Generate an output image that is identical to the reference in every respect "
            f"except that the person is now wearing a {garment_description}.\n\n"
            "STRICT RULES — obey every rule without exception:\n"
            "1. IDENTITY PRESERVATION: The person's face, skin tone, eye colour, "
            "facial structure, hair colour, and hair style must be an exact match "
            "to the reference photograph. Do not alter the face in any way.\n"
            "2. POSE & BODY: Maintain the same body shape, proportions, stance, "
            "and pose as shown in the reference.\n"
            "3. BACKGROUND: Reproduce the background, environment, lighting direction, "
            "and colour temperature exactly as in the reference.\n"
            f"4. GARMENT — what to show: Dress the person in a {garment_description}. "
            "The garment must drape and fit naturally on the body, obey fabric physics, "
            "and be lit consistently with the rest of the scene.\n"
            f"5. GARMENT — colour and style fidelity: The {garment_description} must "
            "retain its precise colour, texture, and silhouette as implied by the name. "
            "Do not substitute a different colour or style.\n"
            "6. SCOPE OF CHANGE: Replace only the clothing that the garment would cover. "
            "Preserve all other visible details (accessories, shoes, etc.) unless they "
            "are naturally concealed by the new garment.\n"
            "FORBIDDEN (never do any of these):\n"
            "- Do not generate a different or composite person.\n"
            "- Do not modify the face, hair, or skin tone.\n"
            "- Do not change or reinterpret the background.\n"
            "- Do not change the garment colour to something not described.\n"
            "- Do not add clothing items that were not requested.\n\n"
            "Output: one single photorealistic image."
        )

        result = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt=prompt,
            reference_images=[person_ref],
            config=EditImageConfig(
                number_of_images=1,
                person_generation='allow_adult',
            ),
        )

        final_ai_path = os.path.join(UPLOAD_FOLDER, 'ai_preview.jpg')
        result.generated_images[0].image.save(location=final_ai_path)
        cache_bust = int(_time.time())
        return jsonify({
            'success': True,
            'preview_url': f'/static/uploads/ai_preview.jpg?v={cache_bust}',
            'garment_used': garment_description,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
