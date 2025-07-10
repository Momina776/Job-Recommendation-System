// Debug flag
const DEBUG = true;

// Debug logging function
function debugLog(message, data = null) {
    if (DEBUG) {
        if (data) {
            console.log(`[Debug] ${message}:`, data);
        } else {
            console.log(`[Debug] ${message}`);
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const elements = {
        form: document.getElementById('applicationForm'),
        submitBtn: document.getElementById('submitApplication'),
        modal: document.getElementById('applyModal'),
        applyBtn: document.getElementById('applyNowButton'),
        loader: document.getElementById('loadingOverlay'),
        coverLetter: document.getElementById('cover_letter')
    };

    // Validate elements exist
    debugLog('Elements found:', {
        form: !!elements.form,
        submitBtn: !!elements.submitBtn,
        modal: !!elements.modal,
        applyBtn: !!elements.applyBtn,
        loader: !!elements.loader,
        coverLetter: !!elements.coverLetter
    });

    let submitInProgress = false;
    const modalInstance = elements.modal ? new bootstrap.Modal(elements.modal) : null;

    // Apply button click handler
    if (elements.applyBtn && modalInstance) {
        elements.applyBtn.addEventListener('click', handleApplyClick);
    }

    // Form submission handler
    if (elements.form) {
        elements.form.addEventListener('submit', handleFormSubmit);
    }

    // Cover letter validation
    if (elements.coverLetter) {
        const debouncedValidate = debounce(validateCoverLetter, 500);
        elements.coverLetter.addEventListener('input', () => debouncedValidate(elements));
    }

    // Event handler functions
    function handleApplyClick(e) {
        e.preventDefault();
        if (submitInProgress || this.disabled) {
            debugLog('Button disabled or submission in progress');
            return;
        }
        modalInstance.show();
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        if (submitInProgress) {
            debugLog('Submission in progress');
            return;
        }

        try {
            submitInProgress = true;
            updateUI('submitting');

            const formData = new FormData(this);
            const response = await sendApplication(formData, this.action);
            
            if (response.success) {
                modalInstance.hide();
                await showSuccess();
                redirectToApplications();
            } else {
                throw new Error(response.message || 'Submission failed');
            }
        } catch (error) {
            debugLog('Submission error:', error);
            showError('Application submission failed. Please try again.');
        } finally {
            submitInProgress = false;
            updateUI('idle');
        }
    }

    // Helper functions
    function updateUI(state) {
        if (state === 'submitting') {
            elements.submitBtn.disabled = true;
            elements.submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
            elements.loader.style.display = 'flex';
        } else {
            elements.submitBtn.disabled = false;
            elements.submitBtn.textContent = 'Submit Application';
            elements.loader.style.display = 'none';
        }
    }

    async function sendApplication(formData, url) {
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    function validateCoverLetter(elements) {
        const { coverLetter, submitBtn } = elements;
        const minLength = 100, maxLength = 5000;
        const length = coverLetter.value.length;
        
        let status = {
            valid: length >= minLength && length <= maxLength,
            message: ''
        };

        if (length < minLength) {
            status.message = `Please write at least ${minLength - length} more characters.`;
        } else if (length > maxLength) {
            status.message = `Please remove ${length - maxLength} characters.`;
        } else {
            status.message = 'Cover letter length is good!';
        }

        updateValidationUI(status, elements);
    }

    function updateValidationUI(status, elements) {
        const { submitBtn } = elements;
        const msgEl = document.querySelector('.validation-message') || createValidationMessage(elements.coverLetter);
        
        submitBtn.disabled = !status.valid;
        msgEl.className = `validation-message small mt-1 text-${status.valid ? 'success' : 'danger'}`;
        msgEl.textContent = status.message;
    }

    function createValidationMessage(input) {
        const msg = document.createElement('div');
        msg.className = 'validation-message small mt-1';
        input.parentElement.appendChild(msg);
        return msg;
    }

    function showSuccess() {
        return Swal.fire({
            title: 'Success!',
            text: 'Your application has been submitted successfully.',
            icon: 'success',
            confirmButtonColor: '#3085d6'
        });
    }

    function showError(message) {
        return Swal.fire({
            title: 'Error',
            text: message,
            icon: 'error',
            confirmButtonColor: '#d33'
        });
    }

    function redirectToApplications() {
        window.location.href = '/my-applications';
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
});
