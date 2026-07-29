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

// --- LIVE DEBUG POPUP FUNCTION ---
function showErrorPopup(message, errorDetails) {
    // Pehle se agar koi popup hai toh hatao
    const existingPopup = document.getElementById('debug-popup');
    if(existingPopup) existingPopup.remove();

    // Naya popup banayein
    const popup = document.createElement('div');
    popup.id = 'debug-popup';
    popup.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 90%; max-width: 400px; background: #161b22; color: #f0f6fc;
        padding: 20px; border-radius: 12px; border: 2px solid #da3633;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8); z-index: 9999;
        font-family: system-ui, sans-serif;
    `;
    
    popup.innerHTML = `
        <h3 style="color: #da3633; margin-top:0;">❌ Error Detected</h3>
        <p style="font-size:0.9rem; background:#0d1117; padding:10px; border-radius:6px; word-break:break-all;">
            <strong>Message:</strong> ${message}<br><br>
            <strong>Details:</strong> <span style="color:#ffa657;">${errorDetails}</span>
        </p>
        <p style="font-size:0.85rem; color:#8b949e; margin-top:10px;">
            💡 <strong>Tip:</strong> Open this website on a <strong>Laptop/PC</strong> using <strong>Google Chrome or Edge</strong>.<br>
            Mobile WebGPU is unstable for this tool.
        </p>
        <button onclick="this.parentElement.remove()" style="margin-top:15px; width:100%; padding:10px; background:#238636; color:white; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">
            Got it, Close
        </button>
    `;
    document.body.appendChild(popup);
}

// 1. Update Status Function
function setStatus(text, type) {
    statusBadge.innerText = text;
    statusBadge.className = 'badge-' + type;
}

// 2. Progress Bar Function
function updateProgress(percent, text) {
    progressContainer.style.display = 'block';
    progressFill.style.width = percent + '%';
    progressText.innerText = text;
    if (percent === 100) {
        setTimeout(() => { progressContainer.style.display = 'none'; }, 1500);
    }
}

// 3. Load Model
async function loadModel() {
    setStatus("Downloading Model...", "working");
    updateProgress(0, "Loading AI...");

    try {
        const ort = window.ort;
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        updateProgress(30, "Initializing...");
        
        session = await ort.InferenceSession.create('static/xlsr.onnx', {
            executionProviders: ['webgpu', 'wasm'] // Pehle WebGPU try karega, nahi to WASM
        });

        updateProgress(100, "Model Ready!");
        setStatus("✅ Model Loaded", "ready");
        imageInput.disabled = false;
        
    } catch (e) {
        updateProgress(0, "Error loading model");
        setStatus("❌ Error", "error");
        console.error("Detailed Error:", e);
        
        // 👇 YAHAN POPUP CALL HO RAHA HAI
        showErrorPopup("Failed to create AI session", e.message || e);
    }
}

// 4. Handle Image Upload
uploadArea.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', function(e) {
    if (!session) { 
        showErrorPopup("Model not loaded", "Please wait, model file 'static/xlsr.onnx' failed to load. Try on PC.");
        return; 
    }
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

    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputSize; tempCanvas.height = inputSize;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, inputSize, inputSize);
    let imageData = tempCtx.getImageData(0, 0, inputSize, inputSize);

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

        const imageDataOutput = ctx.createImageData(outputSize, outputSize);
        for (let i = 0; i < outputSize * outputSize; i++) {
            imageDataOutput.data[i * 4] = Math.min(255, outputData[i]);
            imageDataOutput.data[i * 4 + 1] = Math.min(255, outputData[i + outputSize * outputSize]);
            imageDataOutput.data[i * 4 + 2] = Math.min(255, outputData[i + 2 * outputSize * outputSize]);
            imageDataOutput.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageDataOutput, 0, 0);

        currentOutputDataURL = outputCanvas.toDataURL('image/png');
        downloadBtn.disabled = false;
        loadingSpinner.classList.add('hidden');
        setStatus("✅ Enhancement Done!", "ready");

    } catch (e) {
        loadingSpinner.classList.add('hidden');
        setStatus("❌ Enhancement Failed", "error");
        showErrorPopup("AI Upscaling Failed", e.message || e);
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

window.onload = loadModel;