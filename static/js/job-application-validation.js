// Job application form validation
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const elements = {
        form: document.getElementById('applicationForm'),
        coverLetter: document.getElementById('cover_letter'),
        charCount: document.getElementById('charCount'),
        minCharNotice: document.getElementById('minCharNotice'),
        placeholderWarning: document.getElementById('placeholderWarning'),
        validationMessages: document.getElementById('validationMessages'),
        submitBtn: document.getElementById('submitApplication'),
        spinner: document.querySelector('.spinner-border'),
        buttonText: document.querySelector('.button-text')
    };

    // Debug logging
    console.log('Form elements initialized:', {
        form: !!elements.form,
        coverLetter: !!elements.coverLetter,
        charCount: !!elements.charCount,
        submitBtn: !!elements.submitBtn
    });

    if (elements.form) {
        elements.form.addEventListener('submit', handleSubmit);
    }

    if (elements.coverLetter) {
        elements.coverLetter.addEventListener('input', debounce(validateForm, 300));
    }

    async function handleSubmit(e) {
        e.preventDefault();
        console.log('Form submission started');

        if (!validateForm()) {
            console.log('Validation failed');
            return;
        }

        updateUIState('submitting');
        console.log('Form action URL:', elements.form.action);

        try {            // Convert FormData to URLSearchParams for x-www-form-urlencoded
            const formData = new FormData(elements.form);
            const urlEncodedData = new URLSearchParams();
            for (const [key, value] of formData) {
                urlEncodedData.append(key, value);
            }
            
            // Log form data for debugging
            console.log('Form data being sent:', {
                url: elements.form.action,
                coverLetterLength: formData.get('cover_letter')?.length
            });

            const response = await fetch(elements.form.action, {
                method: 'POST',
                body: urlEncodedData.toString(),
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'same-origin'
            });

            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Server response:', data);

            if (!response.ok) {
                throw new Error(data.message || 'Application submission failed');
            }

            await Swal.fire({
                title: 'Success!',
                text: data.message || 'Your application has been submitted successfully!',
                icon: 'success',
                confirmButtonText: 'OK'
            });

            if (data.redirectUrl) {
                window.location.href = data.redirectUrl;
            }

        } catch (error) {
            console.error('Submission error:', error);
            showErrors([error.message]);
        } finally {
            updateUIState('idle');
        }
    }

    function validateForm() {
        const coverLetter = elements.coverLetter.value.trim();
        const errors = [];

        // Update character count
        elements.charCount.textContent = coverLetter.length;

        // Check minimum length
        if (coverLetter.length < 100) {
            errors.push(`Please write at least ${100 - coverLetter.length} more characters`);
            elements.minCharNotice.classList.remove('d-none');
        } else {
            elements.minCharNotice.classList.add('d-none');
        }

        // Check for placeholders
        const placeholders = ['[Your Field/Expertise]', '[Position]', '[Company]'];
        const hasPlaceholders = placeholders.some(p => coverLetter.includes(p));
        elements.placeholderWarning.classList.toggle('d-none', !hasPlaceholders);
        
        if (hasPlaceholders) {
            errors.push('Please replace all placeholder text');
        }

        if (errors.length > 0) {
            showErrors(errors);
            elements.submitBtn.disabled = true;
            return false;
        }

        hideErrors();
        elements.submitBtn.disabled = false;
        return true;
    }

    function showErrors(errors) {
        elements.validationMessages.classList.remove('d-none');
        elements.validationMessages.innerHTML = errors.map(err => `<div>${err}</div>`).join('');
    }

    function hideErrors() {
        elements.validationMessages.classList.add('d-none');
        elements.validationMessages.innerHTML = '';
    }

    function updateUIState(state) {
        const isSubmitting = state === 'submitting';
        elements.submitBtn.disabled = isSubmitting;
        elements.spinner.classList.toggle('d-none', !isSubmitting);
        elements.buttonText.textContent = isSubmitting ? 'Submitting...' : 'Submit Application';
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

    // Initialize validation
    validateForm();
});
