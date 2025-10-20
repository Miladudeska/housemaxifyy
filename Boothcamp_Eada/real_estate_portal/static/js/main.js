// Main JS for DreamNest portal

document.addEventListener('DOMContentLoaded', function () {
  // Nav toggle
  const toggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');
  if (toggle && navLinks) {
    toggle.addEventListener('click', function () {
      const expanded = this.getAttribute('aria-expanded') === 'true';
      this.setAttribute('aria-expanded', !expanded);
      navLinks.classList.toggle('open');
    });
  }

  // Simple reveal animation on load
  const hero = document.querySelector('.hero-content');
  if (hero) {
    hero.classList.add('reveal');
  }

  const cards = document.querySelectorAll('.feature-card');
  cards.forEach((c, i) => {
    setTimeout(() => c.classList.add('reveal'), 180 * i);
  });

  /* Modal quick estimate */
  const modal = document.getElementById('estimateModal');
  const modalOpenBtns = document.querySelectorAll('.cta-button');
  const modalClose = modal && modal.querySelector('.modal-close');
  const mSubmit = document.getElementById('m_submit');

  // open modal when any CTA with data-open-modal is clicked (or the first CTA on page)
  modalOpenBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      // If it is a link to the form, let the default behaviour occur
      if (btn.tagName.toLowerCase() === 'a') return;
      e.preventDefault();
      if (modal) {
        modal.setAttribute('aria-hidden', 'false');
        modal.classList.add('open');
      }
    });
  });

  if (modalClose) {
    modalClose.addEventListener('click', () => {
      modal.setAttribute('aria-hidden', 'true');
      modal.classList.remove('open');
    });
  }

  if (mSubmit) {
    mSubmit.addEventListener('click', async (e) => {
      e.preventDefault();
      const sqft = document.getElementById('m_sqft').value;
      const bed = document.getElementById('m_bed').value;
      const bath = document.getElementById('m_bath').value;

      const payload = { sqft_living: sqft, bedrooms: bed, bathrooms: bath };
      try {
        const resp = await fetch('/api/predict', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });
        const data = await resp.json();
        const result = document.getElementById('m_result');
        if (resp.ok && data.price) {
          result.style.display = 'block';
          result.innerHTML = `<strong>Estimate:</strong> $${Math.round(data.low).toLocaleString()} - $${Math.round(data.high).toLocaleString()}<br><small>Point estimate: $${Math.round(data.price).toLocaleString()}</small>`;
        } else {
          result.style.display = 'block';
          result.innerText = data.error || 'Prediction error';
        }
      } catch (err) {
        const result = document.getElementById('m_result');
        result.style.display = 'block';
        result.innerText = 'Network error';
      }
    });
  }
});
