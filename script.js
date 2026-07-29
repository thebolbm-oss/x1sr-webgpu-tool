
let session = null;
let currentOutputDataURL = null;

// 1. Model Load Function (With WebGPU priority, else WASM/CPU)
async function loadModel() {
    updateStatus("Loading AI Model... Please wait.", 'loading');
    try {
        const ort = window.ort;
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        // Try WebGPU first. If fails, fallback to WASM (CPU) automatically.
        session = await ort.InferenceSession.create('xlsr.onnx', {
            executionProviders: ['webgpu', 'wasm'] 
        });
        
        console.log("Model Loaded successfully on: ", session.executionProvider);
        updateStatus("✅ Model Ready! Upload an image.", 'idle');
        document.getElementById('imageInput').disabled = false;
    } catch (e) {
        updateStatus("❌ Error: " + e.message, 'error');
        console.error(e);
    }
}

// 2. UI Status updater
function updateStatus(message, type) {
    const bar = document.getElementById('status-bar');
    bar.innerText = message;
    bar.className = 'status-' + type;
}

// 3. Image Upload Handler
document.getElementById('uploadArea').addEventListener('click', () => {
    document.getElementById('imageInput').click();
});

document.getElementById('imageInput').addEventListener('change', async function(e) {
    if (!session) { alert("Model still loading! Wait a second."); return; }
    const file = e.target.files[0];
    if (!file) return;

    const img = document.getElementById('originalImage');
    img.src = URL.createObjectURL(file);
    document.getElementById('downloadBtn').disabled = true;
    currentOutputDataURL = null;

    img.onload = async () => {
        updateStatus("⚡ Upscaling with WebGPU...", 'loading');
        document.getElementById('loading-spinner').classList.remove('spinner-hidden');
        await upscaleImage(img);
    };
});

// 4. Main Upscaling Logic (Optimized for w8a8 quantized XLSR)
async function upscaleImage(img) {
    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');

    const inputSize = 128; // XLSR 3x requires 128x128 input
    const scale = 3;
    const outputSize = inputSize * scale; // 384x384

    canvas.width = outputSize;
    canvas.height = outputSize;

    // 1. Prepare Input (Resize to 128x128)
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputSize;
    tempCanvas.height = inputSize;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, inputSize, inputSize);
    let imageData = tempCtx.getImageData(0, 0, inputSize, inputSize);

    // 2. Convert to Tensor (Uint8 for w8a8 model)
    const { data, width, height } = imageData;
    const dataTensor = new Uint8Array(3 * width * height);
    
    for (let i = 0; i < width * height; i++) {
        dataTensor[i] = data[i * 4];                 // R (0-255)
        dataTensor[i + width * height] = data[i * 4 + 1]; // G
        dataTensor[i + 2 * width * height] = data[i * 4 + 2]; // B
    }

    const tensor = new ort.Tensor('uint8', dataTensor, [1, 3, height, width]);

    // 3. Run AI Model
    try {
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        const outputData = results.output.data;

        // 4. Convert Output Tensor to ImageData
        const imageDataOutput = ctx.createImageData(outputSize, outputSize);
        for (let i = 0; i < outputSize * outputSize; i++) {
            imageDataOutput.data[i * 4] = Math.min(255, outputData[i]);         // R
            imageDataOutput.data[i * 4 + 1] = Math.min(255, outputData[i + outputSize * outputSize]); // G
            imageDataOutput.data[i * 4 + 2] = Math.min(255, outputData[i + 2 * outputSize * outputSize]); // B
            imageDataOutput.data[i * 4 + 3] = 255; // Alpha
        }
        ctx.putImageData(imageDataOutput, 0, 0);
        
        // 5. Enable Download & Update UI
        currentOutputDataURL = canvas.toDataURL('image/png');
        document.getElementById('downloadBtn').disabled = false;
        document.getElementById('loading-spinner').classList.add('spinner-hidden');
        updateStatus("✅ Upscale Complete! Click Download.", 'success');

    } catch (e) {
        document.getElementById('loading-spinner').classList.add('spinner-hidden');
        updateStatus("❌ Upscale Failed: " + e.message, 'error');
    }
}

// 5. Download Button Logic
document.getElementById('downloadBtn').addEventListener('click', () => {
    if (currentOutputDataURL) {
        const link = document.createElement('a');
        link.download = 'upscaled_3x_image.png';
        link.href = currentOutputDataURL;
        link.click();
    }
});

// Start the app
window.onload = loadModel;