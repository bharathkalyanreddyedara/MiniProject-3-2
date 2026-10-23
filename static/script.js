const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');

let selectedFile = null;

// Click to upload
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// File selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

function handleFile(file) {
    selectedFile = file;
    
    // Update UI
    const uploadTitle = uploadBox.querySelector('.upload-title');
    const uploadText = uploadBox.querySelector('.upload-text');
    
    uploadTitle.textContent = 'File Selected';
    uploadText.textContent = file.name;
    uploadBox.style.borderColor = 'var(--primary)';
    uploadBox.style.background = '#eff6ff';
    
    analyzeBtn.disabled = false;
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    // Show loader
    loader.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error analyzing image: ' + error.message);
    } finally {
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    // Update stats
    document.getElementById('urlCount').textContent = data.urls_found;
    
    const benignCount = data.results.filter(r => r.prediction === 'BENIGN').length;
    const maliciousCount = data.results.filter(r => r.prediction === 'MALICIOUS').length;
    
    document.getElementById('benignCount').textContent = benignCount;
    document.getElementById('maliciousCount').textContent = maliciousCount;
    
    // Hidden text
    let displayText = '';
    if (data.stego_text && data.stego_text !== 'No steganographic data found') {
        displayText += `📦 Steganographic Data:\n${data.stego_text}\n\n`;
    }
    if (data.ocr_text && data.ocr_text !== 'No visible text detected') {
        displayText += `👁️ Visible Text (OCR):\n${data.ocr_text}\n\n`;
    }
    if (data.qr_codes && data.qr_codes.length > 0) {
        displayText += `📱 QR Codes Found (${data.qr_codes.length}):\n`;
        data.qr_codes.forEach((qr, index) => {
            displayText += `${index + 1}. [${qr.type}] ${qr.data}\n`;
        });
    }
    
    if (!displayText) {
        displayText = 'No text data found in image';
    }
    
    document.getElementById('hiddenText').textContent = displayText;
    
    // URL results
    const urlResultsDiv = document.getElementById('urlResults');
    urlResultsDiv.innerHTML = '';
    
    if (data.results.length === 0) {
        urlResultsDiv.innerHTML = `
            <div style="text-align: center; padding: 32px; color: var(--text-secondary);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin: 0 auto 16px; opacity: 0.5;">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="16" x2="12" y2="12"/>
                    <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                <p>No URLs detected in the image</p>
            </div>
        `;
    } else {
        data.results.forEach(result => {
            const urlItem = document.createElement('div');
            urlItem.className = `url-item ${result.prediction.toLowerCase()}`;
            
            const icon = result.prediction === 'MALICIOUS' 
                ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
                : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
            
            const sourceIcon = getSourceIcon(result.source);
            
            urlItem.innerHTML = `
                <div class="url-header">
                    <div class="url-source">${sourceIcon} ${result.source}</div>
                </div>
                <div class="url-text">${escapeHtml(result.url)}</div>
                <div class="url-prediction">
                    <span class="badge ${result.prediction.toLowerCase()}">
                        ${icon}
                        ${result.prediction === 'MALICIOUS' ? 'Malicious' : 'Benign'}
                    </span>
                    
                </div>
            `;
            
            urlResultsDiv.appendChild(urlItem);
        });
    }
    
    results.classList.remove('hidden');
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getSourceIcon(source) {
    if (source.includes('Steganography')) {
        return '🔐';
    } else if (source.includes('QR Code')) {
        return '📱';
    } else if (source.includes('OCR')) {
        return '👁️';
    }
    return '📄';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

