// ==UserScript==
// @name         JobBot Copilot Assistant
// @namespace    http://localhost:5000/
// @version      1.0.0
// @description  1-Click In-Browser Autofill Assistant for Workday, Greenhouse, Lever, and Ashby
// @author       JobBot
// @match        https://*.myworkdayjobs.com/*
// @match        https://boards.greenhouse.io/*
// @match        https://job-boards.greenhouse.io/*
// @match        https://jobs.lever.co/*
// @match        https://jobs.ashbyhq.com/*
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setClipboard
// ==/UserScript==

(function () {
    'use strict';

    if (window.__JOBBOT_COPILOT_INITIALIZED__) {
        console.log('[JobBot Copilot] Already loaded.');
        if (window.__JOBBOT_COPILOT_OPEN__) window.__JOBBOT_COPILOT_OPEN__();
        return;
    }
    window.__JOBBOT_COPILOT_INITIALIZED__ = true;

    const API_BASE = 'http://localhost:5000';
    let currentPackage = null;
    let filledCount = 0;
    let blankCount = 0;

    // Load persisted state if navigating multi-step wizard
    try {
        const saved = sessionStorage.getItem('jobbot_copilot_pkg');
        if (saved) {
            currentPackage = JSON.parse(saved);
        }
    } catch (e) {}

    // --- Helper: Dispatch input events so framework bindings update ---
    function setElementValue(el, value) {
        if (!el || value === undefined || value === null) return false;
        const strVal = String(value);

        if (el.tagName === 'SELECT') {
            let matched = false;
            const normTarget = strVal.trim().toLowerCase();
            for (let i = 0; i < el.options.length; i++) {
                const optText = el.options[i].text.trim().toLowerCase();
                const optVal = el.options[i].value.trim().toLowerCase();
                if (optText === normTarget || optVal === normTarget || (normTarget && optText.includes(normTarget))) {
                    el.selectedIndex = i;
                    matched = true;
                    break;
                }
            }
            if (!matched && (normTarget === 'yes' || normTarget === 'no')) {
                for (let i = 0; i < el.options.length; i++) {
                    const optText = el.options[i].text.trim().toLowerCase();
                    if (optText.startsWith(normTarget)) {
                        el.selectedIndex = i;
                        matched = true;
                        break;
                    }
                }
            }
            el.dispatchEvent(new Event('change', { bubbles: true }));
            el.dispatchEvent(new Event('input', { bubbles: true }));
            markFilled(el);
            return true;
        }

        if (el.type === 'checkbox' || el.type === 'radio') {
            const isTrue = ['yes', 'true', '1', 'authorized'].includes(strVal.trim().toLowerCase());
            if (el.type === 'checkbox') {
                el.checked = isTrue;
            } else if (el.type === 'radio') {
                const val = (el.value || '').toLowerCase();
                const label = getLabelText(el).toLowerCase();
                if ((isTrue && (val === 'yes' || val === 'true' || label.includes('yes'))) ||
                    (!isTrue && (val === 'no' || val === 'false' || label.includes('no')))) {
                    el.checked = true;
                }
            }
            el.dispatchEvent(new Event('change', { bubbles: true }));
            el.dispatchEvent(new Event('click', { bubbles: true }));
            markFilled(el);
            return true;
        }

        // Text, email, tel, textarea
        const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const setVal = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
        if (setVal) {
            setVal.call(el, strVal);
        } else {
            el.value = strVal;
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        el.dispatchEvent(new Event('blur', { bubbles: true }));
        markFilled(el);
        return true;
    }

    function markFilled(el) {
        try {
            el.style.transition = 'box-shadow 0.3s ease, border-color 0.3s ease';
            el.style.borderColor = '#10b981';
            el.style.boxShadow = '0 0 0 2px rgba(16, 185, 129, 0.25)';
            el.setAttribute('data-jobbot-filled', 'true');
        } catch (e) {}
    }

    function markAIFilled(el, source) {
        try {
            el.style.transition = 'box-shadow 0.3s ease, border-color 0.3s ease';
            el.style.borderColor = '#8b5cf6';
            el.style.boxShadow = '0 0 0 2px rgba(139, 92, 246, 0.35)';
            el.setAttribute('data-jobbot-filled', 'true');
            el.setAttribute('title', `AI Answered via ${source || 'JobBot'}`);
        } catch (e) {}
    }

    function queryAnswerQuestion(question, jobId, options, qtype) {
        return new Promise((resolve) => {
            const url = `${API_BASE}/api/copilot/answer-question`;
            const payload = {
                question: question,
                job_id: jobId,
                options: options || [],
                qtype: qtype || 'text',
            };

            if (typeof GM_xmlhttpRequest !== 'undefined') {
                GM_xmlhttpRequest({
                    method: 'POST',
                    url: url,
                    headers: { 'Content-Type': 'application/json' },
                    data: JSON.stringify(payload),
                    onload: (res) => {
                        try {
                            resolve(JSON.parse(res.responseText));
                        } catch (e) {
                            resolve(null);
                        }
                    },
                    onerror: () => resolve(null),
                });
            } else {
                fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                })
                    .then(r => r.json())
                    .then(resolve)
                    .catch(() => resolve(null));
            }
        });
    }

    async function answerUnmatchedFields(pkg) {
        const jobId = pkg ? pkg.job_id : null;
        const inputs = Array.from(document.querySelectorAll('input, textarea, select')).filter(el => {
            if (el.type === 'hidden' || el.type === 'submit' || el.type === 'button' || el.type === 'reset') return false;
            if (el.disabled || el.readOnly) return false;
            if (el.closest('#jobbot-copilot-badge')) return false;
            if (el.getAttribute('data-jobbot-filled') === 'true') return false;
            if (el.value && el.value.trim().length > 0) return false;
            return true;
        });

        if (inputs.length === 0) {
            showToast('No empty fields remaining!', 'info');
            return;
        }

        showToast(`AI generating answers for ${inputs.length} field(s)...`, 'info');
        let aiFilled = 0;

        for (const el of inputs) {
            const labelText = getLabelText(el);
            if (!labelText || labelText.length < 3) continue;

            let options = [];
            if (el.tagName === 'SELECT') {
                options = Array.from(el.options).map(o => o.text.trim()).filter(Boolean);
            }

            try {
                const res = await queryAnswerQuestion(labelText, jobId, options, el.tagName === 'TEXTAREA' ? 'textarea' : 'text');
                if (res && res.ok && res.answer) {
                    const filled = setElementValue(el, res.answer);
                    if (filled) {
                        aiFilled++;
                        filledCount++;
                        markAIFilled(el, res.source);
                    }
                }
            } catch (err) {
                console.warn('[JobBot Copilot] Failed to get AI answer for field:', labelText, err);
            }
        }

        updateBadgeUI();
        if (aiFilled > 0) {
            showToast(`AI answered ${aiFilled} field(s)!`, 'success');
        } else {
            showToast('No additional fields could be answered by AI.', 'info');
        }
    }

    // --- Helper: Extract Field Identifiers ---
    function getLabelText(el) {
        let label = '';
        if (el.id) {
            const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
            if (lbl) label += ' ' + lbl.innerText;
        }
        const parentLabel = el.closest('label');
        if (parentLabel) label += ' ' + parentLabel.innerText;

        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) label += ' ' + ariaLabel;

        const ariaLabelledby = el.getAttribute('aria-labelledby');
        if (ariaLabelledby) {
            ariaLabelledby.split(/\s+/).forEach(id => {
                const target = document.getElementById(id);
                if (target) label += ' ' + target.innerText;
            });
        }

        const placeholder = el.getAttribute('placeholder');
        if (placeholder) label += ' ' + placeholder;

        const autoId = el.getAttribute('data-automation-id');
        if (autoId) label += ' ' + autoId;

        const name = el.getAttribute('name');
        if (name) label += ' ' + name;

        // Preceding sibling or heading in field group
        const group = el.closest('[data-automation-id], .form-group, .field, fieldset');
        if (group) {
            const legend = group.querySelector('legend, h3, h4, label, .field-label');
            if (legend && legend !== parentLabel) label += ' ' + legend.innerText;
        }

        return label.replace(/\s+/g, ' ').trim().toLowerCase();
    }

    // --- Autofill Engine ---
    function runAutofill(pkg) {
        if (!pkg || !pkg.applicant) {
            showToast('No application package loaded!', 'error');
            return;
        }

        const applicant = pkg.applicant;
        const answers = pkg.screening_answers || {};
        filledCount = 0;
        blankCount = 0;

        const inputs = Array.from(document.querySelectorAll('input, textarea, select')).filter(el => {
            if (el.type === 'hidden' || el.type === 'submit' || el.type === 'button' || el.type === 'reset') return false;
            if (el.disabled || el.readOnly) return false;
            if (el.closest('#jobbot-copilot-badge')) return false;
            return true;
        });

        inputs.forEach(el => {
            const text = getLabelText(el);
            const autoId = (el.getAttribute('data-automation-id') || '').toLowerCase();
            const name = (el.getAttribute('name') || '').toLowerCase();
            const id = (el.getAttribute('id') || '').toLowerCase();
            let filled = false;

            // 1. Workday specific data-automation-id overrides
            if (autoId.includes('legalnamesection_firstname') || autoId === 'firstname') {
                filled = setElementValue(el, applicant.first_name);
            } else if (autoId.includes('legalnamesection_lastname') || autoId === 'lastname') {
                filled = setElementValue(el, applicant.last_name);
            } else if (autoId.includes('addresssection_addressline1') || autoId === 'addressline1') {
                filled = setElementValue(el, applicant.address);
            } else if (autoId.includes('addresssection_city') || autoId === 'city') {
                filled = setElementValue(el, applicant.city);
            } else if (autoId.includes('addresssection_postalcode') || autoId === 'postalcode') {
                filled = setElementValue(el, applicant.zip);
            } else if (autoId.includes('addresssection_countryregion') || autoId === 'countryregion') {
                filled = setElementValue(el, applicant.state);
            } else if (autoId.includes('phone-number') || autoId === 'phonenumber') {
                filled = setElementValue(el, applicant.phone);
            } else if (autoId === 'email' || autoId === 'emailaddress') {
                filled = setElementValue(el, applicant.email);
            }

            if (filled) {
                filledCount++;
                return;
            }

            // 2. Identity field matcher by label / name / id
            if (/\b(first|given)\s*name\b/i.test(text) || name.includes('first_name') || id.includes('first_name')) {
                filled = setElementValue(el, applicant.first_name);
            } else if (/\b(last|family|sur)\s*name\b/i.test(text) || name.includes('last_name') || id.includes('last_name')) {
                filled = setElementValue(el, applicant.last_name);
            } else if (/\b(full\s*name|candidate\s*name)\b/i.test(text) || (name === 'name' && !name.includes('company'))) {
                filled = setElementValue(el, applicant.full_name || `${applicant.first_name} ${applicant.last_name}`.trim());
            } else if (/\b(e-?mail)\b/i.test(text) || el.type === 'email' || name.includes('email')) {
                filled = setElementValue(el, applicant.email);
            } else if (/\b(phone|mobile|cell|telephone|contact\s*number)\b/i.test(text) || el.type === 'tel' || name.includes('phone')) {
                filled = setElementValue(el, applicant.phone);
            } else if (/linkedin/i.test(text) || name.includes('linkedin') || text.includes('urls[linkedin]')) {
                filled = setElementValue(el, applicant.linkedin);
            } else if (/github/i.test(text) || name.includes('github') || text.includes('urls[github]')) {
                filled = setElementValue(el, applicant.github);
            } else if (/portfolio|personal\s*website|website|orcid/i.test(text) || name.includes('portfolio') || text.includes('urls[portfolio]')) {
                filled = setElementValue(el, applicant.portfolio || applicant.github || applicant.linkedin);
            } else if (/street\s*address|address\s*line\s*1|residential\s*address/i.test(text)) {
                filled = setElementValue(el, applicant.address);
            } else if (/\b(city|town)\b/i.test(text) && !text.includes('state')) {
                filled = setElementValue(el, applicant.city);
            } else if (/\b(state|province|region)\b/i.test(text)) {
                filled = setElementValue(el, applicant.state);
            } else if (/\b(zip|postal\s*code|postcode)\b/i.test(text)) {
                filled = setElementValue(el, applicant.zip);
            } else if (/current\s*(title|headline|role)/i.test(text)) {
                filled = setElementValue(el, applicant.current_title);
            } else if (/cover\s*letter|letter\s*of\s*interest|note\s*to\s*hiring/i.test(text)) {
                if (pkg.cover_letter_text) {
                    filled = setElementValue(el, pkg.cover_letter_text);
                }
            }

            if (filled) {
                filledCount++;
                return;
            }

            // 3. Screening Questions Matching
            // Work authorization
            if (/authoriz|eligible to work|authorized to work|legal right to work/i.test(text)) {
                filled = setElementValue(el, answers.work_authorized || 'Yes');
            }
            // Visa sponsorship
            else if (/sponsorship|require.*visa|visa.*sponsor/i.test(text)) {
                filled = setElementValue(el, answers.requires_sponsorship || 'No');
            }
            // Salary expectation
            else if (/salary|compensation|desired pay|expected rate/i.test(text)) {
                if (answers.salary_expectation) {
                    filled = setElementValue(el, answers.salary_expectation);
                }
            }
            // Earliest start date / notice
            else if (/start date|earliest.*start|available.*start|notice period/i.test(text)) {
                if (answers.earliest_start_date) {
                    filled = setElementValue(el, answers.earliest_start_date);
                }
            }
            // Relocation
            else if (/relocate|relocation/i.test(text)) {
                if (answers.willing_relocate) {
                    filled = setElementValue(el, answers.willing_relocate);
                }
            }
            // Onsite / hybrid
            else if (/on-?site|hybrid|commute|willing to work on/i.test(text)) {
                filled = setElementValue(el, answers.willing_onsite || 'Yes');
            }
            // How heard / source
            else if (/how did you hear|referral source|where did you hear/i.test(text)) {
                filled = setElementValue(el, answers.how_heard || 'Company careers site');
            }

            if (filled) {
                filledCount++;
                return;
            }

            // 4. Custom screening answers lookup from package
            for (const [qText, qAns] of Object.entries(answers)) {
                if (!qAns) continue;
                const normQ = qText.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim();
                const normField = text.replace(/[^a-z0-9 ]/g, '').trim();
                if (normField && (normField.includes(normQ) || normQ.includes(normField))) {
                    filled = setElementValue(el, qAns);
                    if (filled) {
                        filledCount++;
                        return;
                    }
                }
            }

            if (el.value === '' && el.required) {
                blankCount++;
            }
        });

        updateBadgeUI();
        showToast(`JobBot Copilot: ${filledCount} fields filled!`, 'success');
    }

    // --- Floating Badge UI ---
    function injectBadge() {
        if (document.getElementById('jobbot-copilot-badge')) return;

        const badge = document.createElement('div');
        badge.id = 'jobbot-copilot-badge';
        badge.innerHTML = `
            <style>
                #jobbot-copilot-badge {
                    position: fixed;
                    bottom: 24px;
                    right: 24px;
                    z-index: 999999;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    font-size: 13px;
                    color: #1e293b;
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
                    width: 320px;
                    overflow: hidden;
                    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                }
                #jobbot-copilot-badge.minimized {
                    width: 150px;
                    height: 42px;
                    border-radius: 24px;
                    cursor: pointer;
                }
                #jobbot-copilot-badge .badge-header {
                    background: linear-gradient(135deg, #1e293b, #0f172a);
                    color: #f8fafc;
                    padding: 10px 14px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    font-weight: 600;
                    user-select: none;
                }
                #jobbot-copilot-badge.minimized .badge-body {
                    display: none;
                }
                #jobbot-copilot-badge .badge-body {
                    padding: 14px;
                }
                #jobbot-copilot-badge .badge-row {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 8px;
                }
                #jobbot-copilot-badge .badge-title {
                    font-weight: 600;
                    color: #0f172a;
                    font-size: 13px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 200px;
                }
                #jobbot-copilot-badge .badge-sub {
                    color: #64748b;
                    font-size: 11px;
                }
                #jobbot-copilot-badge .stats-pill {
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 9999px;
                    font-size: 11px;
                    font-weight: 500;
                    background: #ecfdf5;
                    color: #065f46;
                    border: 1px solid #a7f3d0;
                }
                #jobbot-copilot-badge button {
                    display: block;
                    width: 100%;
                    padding: 8px 12px;
                    margin-top: 8px;
                    font-size: 12px;
                    font-weight: 600;
                    border-radius: 8px;
                    border: none;
                    cursor: pointer;
                    transition: background 0.15s ease;
                }
                #jobbot-copilot-btn-fill {
                    background: #2563eb;
                    color: #ffffff;
                }
                #jobbot-copilot-btn-fill:hover {
                    background: #1d4ed8;
                }
                #jobbot-copilot-btn-review {
                    background: #10b981;
                    color: #ffffff;
                }
                #jobbot-copilot-btn-review:hover {
                    background: #059669;
                }
                #jobbot-copilot-btn-copy-resume {
                    background: #f1f5f9;
                    color: #334155;
                    border: 1px solid #cbd5e1 !important;
                }
                #jobbot-copilot-btn-copy-resume:hover {
                    background: #e2e8f0;
                }
                #jobbot-copilot-toast {
                    position: fixed;
                    bottom: 80px;
                    right: 24px;
                    background: #0f172a;
                    color: #fff;
                    padding: 8px 14px;
                    border-radius: 8px;
                    font-size: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 1000000;
                    opacity: 0;
                    transform: translateY(10px);
                    transition: all 0.2s ease;
                    pointer-events: none;
                }
                #jobbot-copilot-toast.show {
                    opacity: 1;
                    transform: translateY(0);
                }
            </style>
            <div class="badge-header" id="jobbot-badge-header">
                <span>⚡ JobBot Copilot</span>
                <span id="jobbot-toggle-btn" style="cursor: pointer; font-size: 14px; padding: 0 4px;">−</span>
            </div>
            <div class="badge-body">
                <div class="badge-row">
                    <div>
                        <div class="badge-title" id="jobbot-job-title">${currentPackage ? currentPackage.company : 'Auto-Detecting...'}</div>
                        <div class="badge-sub" id="jobbot-job-sub">${currentPackage ? (currentPackage.title || 'Job #' + currentPackage.job_id) : 'Click Load or Autofill'}</div>
                    </div>
                    <span class="stats-pill" id="jobbot-stats">Ready</span>
                </div>
                <div style="display: flex; gap: 6px; margin-top: 10px;">
                    <input type="text" id="jobbot-job-id-input" placeholder="Job ID (optional)" style="width: 65%; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 11px;" value="${currentPackage ? currentPackage.job_id : ''}" />
                    <button id="jobbot-btn-fetch" style="width: 35%; margin-top: 0; padding: 5px 8px; background: #475569; color: #fff;">Load</button>
                </div>
                <button id="jobbot-copilot-btn-fill">🚀 Autofill Page</button>
                <button id="jobbot-copilot-btn-review">📋 Review & Submit</button>
                <button id="jobbot-copilot-btn-copy-resume">📄 Copy Resume Path</button>
            </div>
        `;
        document.body.appendChild(badge);

        // Toast container
        const toast = document.createElement('div');
        toast.id = 'jobbot-copilot-toast';
        document.body.appendChild(toast);

        // Event listeners
        const toggleBtn = document.getElementById('jobbot-toggle-btn');
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            badge.classList.toggle('minimized');
            toggleBtn.innerText = badge.classList.contains('minimized') ? '+' : '−';
        });

        badge.addEventListener('click', () => {
            if (badge.classList.contains('minimized')) {
                badge.classList.remove('minimized');
                toggleBtn.innerText = '−';
            }
        });

        document.getElementById('jobbot-btn-fetch').addEventListener('click', () => {
            const idVal = document.getElementById('jobbot-job-id-input').value.trim();
            fetchPackage(idVal);
        });

        document.getElementById('jobbot-copilot-btn-fill').addEventListener('click', () => {
            if (!currentPackage) {
                const idVal = document.getElementById('jobbot-job-id-input').value.trim();
                fetchPackage(idVal, () => runAutofill(currentPackage));
            } else {
                runAutofill(currentPackage);
            }
        });

        document.getElementById('jobbot-copilot-btn-ai-answer').addEventListener('click', () => {
            answerUnmatchedFields(currentPackage);
        });

        document.getElementById('jobbot-copilot-btn-copy-cover').addEventListener('click', () => {
            if (currentPackage && currentPackage.cover_letter_text) {
                copyToClipboard(currentPackage.cover_letter_text);
                showToast('Cover letter copied to clipboard!', 'success');
            } else {
                showToast('No cover letter text in package.', 'error');
            }
        });

        document.getElementById('jobbot-copilot-btn-review').addEventListener('click', () => {
            reviewAndHighlightSubmit();
        });

        document.getElementById('jobbot-copilot-btn-copy-resume').addEventListener('click', () => {
            if (currentPackage && currentPackage.resume_path) {
                copyToClipboard(currentPackage.resume_path);
                showToast('Resume path copied! Paste into file dialog.', 'success');
            } else {
                showToast('No tailored resume path in package.', 'error');
            }
        });

        window.__JOBBOT_COPILOT_OPEN__ = () => {
            badge.classList.remove('minimized');
            toggleBtn.innerText = '−';
        };
    }

    function updateBadgeUI() {
        const titleEl = document.getElementById('jobbot-job-title');
        const subEl = document.getElementById('jobbot-job-sub');
        const statsEl = document.getElementById('jobbot-stats');
        if (!titleEl || !statsEl) return;

        if (currentPackage) {
            titleEl.innerText = currentPackage.company || 'Job Application';
            subEl.innerText = currentPackage.title || `Job #${currentPackage.job_id}`;
            statsEl.innerText = `${filledCount} filled`;
        }
    }

    function showToast(msg, type = 'info') {
        const toast = document.getElementById('jobbot-copilot-toast');
        if (!toast) return;
        toast.innerText = msg;
        toast.style.background = type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#0f172a';
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 3500);
    }

    function copyToClipboard(text) {
        if (typeof GM_setClipboard !== 'undefined') {
            GM_setClipboard(text);
        } else if (navigator.clipboard) {
            navigator.clipboard.writeText(text);
        } else {
            const input = document.createElement('textarea');
            input.value = text;
            document.body.appendChild(input);
            input.select();
            document.execCommand('copy');
            document.body.removeChild(input);
        }
    }

    function reviewAndHighlightSubmit() {
        // Find submit button
        const submitBtn = Array.from(document.querySelectorAll('button, input[type="submit"]')).find(el => {
            const txt = (el.innerText || el.value || '').toLowerCase();
            return txt.includes('submit') || txt.includes('apply') || txt.includes('review');
        });

        if (submitBtn) {
            submitBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
            submitBtn.style.transition = 'all 0.5s ease';
            submitBtn.style.outline = '4px solid #10b981';
            submitBtn.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.6)';
            submitBtn.focus();
            showToast('Form ready! Review fields and click Submit.', 'success');
        } else {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            showToast('Scrolled to bottom of page for review.', 'info');
        }
    }

    // --- Package Fetching ---
    function fetchPackage(jobId, onDone) {
        const idToUse = jobId || window.__JOBBOT_JOB_ID__ || '';
        let url = `${API_BASE}/api/copilot-package`;
        if (idToUse) {
            url = `${API_BASE}/api/job/${encodeURIComponent(idToUse)}/copilot-package`;
        } else {
            url += `?url=${encodeURIComponent(window.location.href)}`;
        }

        const handleSuccess = (data) => {
            currentPackage = data;
            try {
                sessionStorage.setItem('jobbot_copilot_pkg', JSON.stringify(data));
            } catch (e) {}
            updateBadgeUI();
            showToast(`Loaded: ${data.company} - ${data.title}`, 'success');
            if (window.__JOBBOT_AUTOFILL_IMMEDIATE__) {
                setTimeout(() => runAutofill(data), 600);
            }
            if (onDone) onDone();
        };

        if (typeof GM_xmlhttpRequest !== 'undefined') {
            GM_xmlhttpRequest({
                method: 'GET',
                url: url,
                onload: (res) => {
                    if (res.status >= 200 && res.status < 300) {
                        handleSuccess(JSON.parse(res.responseText));
                    } else {
                        showToast(`Server returned status ${res.status}. Is JobBot running?`, 'error');
                    }
                },
                onerror: () => showToast('Failed to connect to JobBot server on http://localhost:5000', 'error')
            });
        } else {
            fetch(url)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.json();
                })
                .then(handleSuccess)
                .catch(err => {
                    console.error('[JobBot Copilot]', err);
                    showToast('Could not load package from localhost:5000. Enter Job ID manually.', 'error');
                });
        }
    }

    // --- Bootstrapping ---
    injectBadge();
    if (currentPackage) {
        updateBadgeUI();
    } else {
        // Try auto-resolving from current page URL
        fetchPackage('', null);
    }
})();
