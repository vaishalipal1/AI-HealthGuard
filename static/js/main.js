// Basic client-side validation for the prediction form
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    const inputs = form.querySelectorAll("input[required]");
    let valid = true;
    inputs.forEach((input) => {
      if (input.value === "" || Number(input.value) < 0) {
        valid = false;
        input.style.borderColor = "#B00020";
      } else {
        input.style.borderColor = "#D6DCE5";
      }
    });
    if (!valid) {
      e.preventDefault();
      alert("Please fill all fields with valid (non-negative) values.");
    }
  });
});
