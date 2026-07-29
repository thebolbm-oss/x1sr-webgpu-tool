let session = null;
let currentOutputDataURL = null;

// UI Elements
const statusBadge = document.getElementById('statusBadge');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const originalImage = document.getElementById('originalImage');
const outputCanvas = document.getElementById('outputCanvas');
const loadingSpinner = document.getElementById('loadingSpinner');
const downloadBtn = document.getElementById('downloadBtn');

// 1. Update Status Function
function setStatus(text, type) {
    statusBadge.innerText = text;
    statusBadge.className = 'badge-' + type;
}

// 2. Progress Bar Function (File size ~140KB, so it will be instant)
function updateProgress(percent, text) {
    progressContainer.style.display = 'block';
    progressFill.style.width = percent + '%';
    progressText.innerText = text;
    if (percent === 100) {
        setTimeout(() => { progressContainer.style.display = 'none'; }, 1500);
    }
}

// 3. Load Model with Progress Tracking
async function loadModel() {
    setStatus("Downloading Model...", "working");
    updateProgress(0, "Loading AI...");

    try {
        const ort = window.ort;
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        // Simulate progress since file is tiny
        updateProgress(30, "Initializing WebGPU...");
        
        // ⚠️ IMPORTANT: Path changed to 'static/xlsr.onnx'
        session = await ort.InferenceSession.create('static/xlsr.onnx', {
            executionProviders: ['webgpu', 'wasm']
        });

        updateProgress(100, "Model Ready!");
        setStatus("✅ Model Loaded", "ready");
        imageInput.disabled = false;
        
    } catch (e) {
        updateProgress(0, "Error loading model");
        setStatus("❌ Error: Check Console", "error");
        console.error("Load Error:", e);
    }
}

// 4. Handle Image Upload
uploadArea.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', function(e) {
    if (!session) { alert("Please wait, model is still loading!"); return; }
    const file = e.target.files[0];
    if (!file) return;

    originalImage.src = URL.createObjectURL(file);
    downloadBtn.disabled = true;
    currentOutputDataURL = null;
    outputCanvas.getContext('2d').clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    originalImage.onload = async () => {
        setStatus("🔮 Enhancing Image...", "working");
        loadingSpinner.classList.remove('hidden');
        await upscaleImage(originalImage);
    };
});

// 5. AI Upscaling Logic
async function upscaleImage(img) {
    const ctx = outputCanvas.getContext('2d');
    const inputSize = 128;
    const scale = 3;
    const outputSize = inputSize * scale;

    outputCanvas.width = outputSize;
    outputCanvas.height = outputSize;

    // Resize to 128x128
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputSize; tempCanvas.height = inputSize;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, inputSize, inputSize);
    let imageData = tempCtx.getImageData(0, 0, inputSize, inputSize);

    // Convert to Uint8 Tensor (for w8a8 model)
    const { data, width, height } = imageData;
    const dataTensor = new Uint8Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
        dataTensor[i] = data[i * 4];
        dataTensor[i + width * height] = data[i * 4 + 1];
        dataTensor[i + 2 * width * height] = data[i * 4 + 2];
    }

    const tensor = new ort.Tensor('uint8', dataTensor, [1, 3, height, width]);

    try {
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        const outputData = results.output.data;

        // Convert output back to Image
        const imageDataOutput = ctx.createImageData(outputSize, outputSize);
        for (let i = 0; i < outputSize * outputSize; i++) {
            imageDataOutput.data[i * 4] = Math.min(255, outputData[i]);
            imageDataOutput.data[i * 4 + 1] = Math.min(255, outputData[i + outputSize * outputSize]);
            imageDataOutput.data[i * 4 + 2] = Math.min(255, outputData[i + 2 * outputSize * outputSize]);
            imageDataOutput.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageDataOutput, 0, 0);

        // Done
        currentOutputDataURL = outputCanvas.toDataURL('image/png');
        downloadBtn.disabled = false;
        loadingSpinner.classList.add('hidden');
        setStatus("✅ Enhancement Done!", "ready");

    } catch (e) {
        loadingSpinner.classList.add('hidden');
        setStatus("❌ Enhancement Failed", "error");
        console.error("Upscale Error:", e);
    }
}

// 6. Download Handler
downloadBtn.addEventListener('click', () => {
    if (currentOutputDataURL) {
        const link = document.createElement('a');
        link.download = 'enhanced_3x.png';
        link.href = currentOutputDataURL;
        link.click();
    }
});

// Start the App
window.onload = loadModel;