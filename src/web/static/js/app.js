/**
 * Price Compare - Frontend Application
 */

const API_BASE = '/api/v1';

// DOM Elements
const textSearchForm = document.getElementById('text-search-form');
const urlSearchForm = document.getElementById('url-search-form');
const imageSearchForm = document.getElementById('image-search-form');
const textQuery = document.getElementById('text-query');
const urlQuery = document.getElementById('url-query');
const imageQuery = document.getElementById('image-query');
const imagePreview = document.getElementById('image-preview');
const enableLiveSearch = document.getElementById('enable-live-search');
const confidenceThreshold = document.getElementById('confidence-threshold');
const thresholdValue = document.getElementById('threshold-value');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const resultCount = document.getElementById('result-count');
const processingTime = document.getElementById('processing-time');
const queryInfo = document.getElementById('query-info');
const resultsGrid = document.getElementById('results-grid');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Tab switching
tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tab = button.dataset.tab;

        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        button.classList.add('active');
        document.getElementById(`${tab}-tab`).classList.add('active');
    });
});

// Confidence threshold slider
confidenceThreshold.addEventListener('input', () => {
    thresholdValue.textContent = `${confidenceThreshold.value}%`;
});

// Image preview
imageQuery.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
});

// Search handlers
textSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textQuery.value.trim();
    if (query) {
        await performSearch(query, 'text');
    }
});

urlSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = urlQuery.value.trim();
    if (query) {
        await performSearch(query, 'url');
    }
});

imageSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = imageQuery.files[0];
    if (file) {
        await performImageSearch(file);
    }
});

// Main search function
async function performSearch(query, inputType) {
    showLoading();
    hideError();
    hideResults();

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                input_type: inputType,
                enable_live_search: enableLiveSearch.checked,
                confidence_threshold: confidenceThreshold.value / 100,
                limit: 20,
            }),
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Image search function
async function performImageSearch(file) {
    showLoading();
    hideError();
    hideResults();

    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('enable_live_search', enableLiveSearch.checked);
        formData.append('confidence_threshold', confidenceThreshold.value / 100);
        formData.append('limit', 20);

        const response = await fetch(`${API_BASE}/search/image`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(data) {
    const results = data.results || [];

    // Update counts
    resultCount.textContent = `${results.length} results found`;
    processingTime.textContent = `${data.processing_time_ms}ms`;

    // Update query info
    const qi = data.query_info || {};
    const props = qi.extracted_properties || {};

    let queryInfoHtml = '<strong>Query:</strong> ';
    if (props.name) queryInfoHtml += props.name;
    if (props.brand) queryInfoHtml += ` | Brand: ${props.brand}`;
    if (props.category) queryInfoHtml += ` | Category: ${props.category}`;
    queryInfoHtml += `<br><strong>Method:</strong> ${qi.search_method || 'N/A'}`;
    if (qi.live_search_triggered) {
        queryInfoHtml += ' | <span style="color: var(--warning-color);">Live search triggered</span>';
    }
    queryInfo.innerHTML = queryInfoHtml;

    // Display results
    resultsGrid.innerHTML = results.map(product => createProductCard(product)).join('');

    showResults();
}

// Create product card HTML
function createProductCard(product) {
    const name = product.name || 'Unknown Product';
    const price = product.price ? formatPrice(product.price, product.currency) : 'N/A';
    const merchant = product.merchant || 'Unknown Merchant';
    const imageUrl = product.image_url;
    const sourceUrl = product.source_url;
    const confidence = product.match_confidence || 0;
    const source = product.match_source || 'database';

    const confidenceClass = confidence >= 0.8 ? 'confidence-high' :
                           confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';

    const imageHtml = imageUrl
        ? `<img src="${imageUrl}" alt="${name}" class="product-image" onerror="this.outerHTML='<div class=\\'product-image placeholder\\'>📦</div>'">`
        : '<div class="product-image placeholder">📦</div>';

    const linkHtml = sourceUrl
        ? `<a href="${sourceUrl}" target="_blank" class="product-link">View Product</a>`
        : '';

    return `
        <div class="product-card">
            ${imageHtml}
            <div class="product-info">
                <h3 class="product-name">${escapeHtml(name)}</h3>
                <p class="product-price">${price}</p>
                <p class="product-merchant">${escapeHtml(merchant)}</p>
                <span class="confidence-badge ${confidenceClass}">
                    ${(confidence * 100).toFixed(0)}% match
                </span>
                <p class="product-source">Source: ${source}</p>
            </div>
            ${linkHtml}
        </div>
    `;
}

// Utility functions
function formatPrice(price, currency = 'USD') {
    const symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'SEK': 'kr',
    };
    const symbol = symbols[currency] || currency + ' ';
    return `${symbol}${price.toFixed(2)}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading() {
    loadingSection.classList.remove('hidden');
}

function hideLoading() {
    loadingSection.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

function hideError() {
    errorSection.classList.add('hidden');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Price Compare initialized');
});
