"""
RunPod Serverless Handler for Hallo
Generates talking head videos from image + audio
"""

import runpod
import base64
import tempfile
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add hallo to path
sys.path.insert(0, '/app/hallo')

# Global model cache
HALLO_MODEL = None


def download_file(url: str, output_path: str) -> bool:
    """Download file from URL"""
    try:
        import requests
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Download error: {e}")
    return False


def get_duration(path: str) -> float:
    """Get video/audio duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0.0


def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio to WAV format (16kHz, mono)"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le',
            output_path
        ], capture_output=True, timeout=60)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False


def run_hallo_inference(image_path: str, audio_path: str, output_path: str) -> bool:
    """Run Hallo inference using command line"""
    try:
        # Convert audio to wav if needed
        wav_path = audio_path
        if not audio_path.lower().endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            if not convert_audio_to_wav(audio_path, wav_path):
                print("Failed to convert audio to WAV")
                return False

        # Hallo inference command
        cmd = [
            'python', '/app/hallo/scripts/inference.py',
            '--source_image', image_path,
            '--driving_audio', wav_path,
            '--output', output_path,
            '--pose_weight', '1.0',
            '--face_weight', '1.0',
            '--lip_weight', '1.0',
            '--face_expand_ratio', '1.2',
        ]

        print(f"Running Hallo: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTHONPATH'] = '/app/hallo:' + env.get('PYTHONPATH', '')

        result = subprocess.run(
            cmd,
            cwd='/app/hallo',
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )

        if result.returncode != 0:
            print(f"Hallo STDOUT: {result.stdout[-1000:]}")
            print(f"Hallo STDERR: {result.stderr[-1000:]}")
            return False

        # Clean up temp wav
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 10000

    except subprocess.TimeoutExpired:
        print("Hallo timeout!")
        return False
    except Exception as e:
        print(f"Hallo error: {e}")
        import traceback
        traceback.print_exc()
        return False


def handler(event):
    """
    RunPod serverless handler

    Input (event['input']):
        - image_base64: Base64 encoded source image (JPG/PNG)
        - audio_base64: Base64 encoded audio file (MP3/WAV)
        - image_url: URL to download image (alternative)
        - audio_url: URL to download audio (alternative)

    Output:
        - video_base64: Base64 encoded output video (MP4)
        - duration: Video duration in seconds
        - error: Error message if failed
    """

    print("=" * 50)
    print("Hallo Handler Started")
    print("=" * 50)

    try:
        job_input = event.get('input', {})

        # Create temp directory
        tmpdir = tempfile.mkdtemp()
        try:
            image_path = os.path.join(tmpdir, 'source.jpg')
            audio_path = os.path.join(tmpdir, 'audio.mp3')
            output_path = os.path.join(tmpdir, 'output.mp4')

            # Get image
            if 'image_base64' in job_input:
                print("Decoding image from base64...")
                image_data = base64.b64decode(job_input['image_base64'])
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            elif 'image_url' in job_input:
                print(f"Downloading image from URL...")
                if not download_file(job_input['image_url'], image_path):
                    return {'error': 'Failed to download image'}
            else:
                return {'error': 'No image provided (image_base64 or image_url required)'}

            # Get audio
            if 'audio_base64' in job_input:
                print("Decoding audio from base64...")
                audio_data = base64.b64decode(job_input['audio_base64'])
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            elif 'audio_url' in job_input:
                print(f"Downloading audio from URL...")
                if not download_file(job_input['audio_url'], audio_path):
                    return {'error': 'Failed to download audio'}
            else:
                return {'error': 'No audio provided (audio_base64 or audio_url required)'}

            image_size = os.path.getsize(image_path)
            audio_size = os.path.getsize(audio_path)
            audio_duration = get_duration(audio_path)

            print(f"Input: image={image_size}B, audio={audio_size}B ({audio_duration:.1f}s)")

            # Run Hallo
            print("Starting Hallo inference...")
            if not run_hallo_inference(image_path, audio_path, output_path):
                return {'error': 'Hallo inference failed'}

            # Check output
            if not os.path.exists(output_path):
                return {'error': 'No output video generated'}

            output_size = os.path.getsize(output_path)
            output_duration = get_duration(output_path)

            print(f"Output: {output_size}B ({output_duration:.1f}s)")

            # Encode output
            with open(output_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')

            print("Success!")
            return {
                'video_base64': video_base64,
                'duration': output_duration,
                'size_bytes': output_size
            }

        finally:
            # Cleanup
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        import traceback
        print(f"Handler error: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# For local testing
def local_test():
    """Test handler locally"""
    import sys
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        audio_path = sys.argv[2]

        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        with open(audio_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        result = handler({
            'input': {
                'image_base64': image_b64,
                'audio_base64': audio_b64
            }
        })

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Duration: {result['duration']}s")
            # Save output
            if len(sys.argv) >= 4:
                with open(sys.argv[3], 'wb') as f:
                    f.write(base64.b64decode(result['video_base64']))
                print(f"Saved to: {sys.argv[3]}")
    else:
        print("Usage: python handler.py <image> <audio> [output.mp4]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        local_test()
    else:
        # RunPod serverless start
        runpod.serverless.start({'handler': handler})
