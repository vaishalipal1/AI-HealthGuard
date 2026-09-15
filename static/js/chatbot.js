function appendMessage(text, sender) {
  const win = document.getElementById("chatWindow");
  const div = document.createElement("div");
  div.className = "chat-msg " + sender;
  div.textContent = text;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  input.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    appendMessage(data.reply || "Sorry, something went wrong.", "bot");
  } catch (err) {
    appendMessage("Network error — please try again.", "bot");
  }
}

function sendQuick(text) {
  document.getElementById("chatInput").value = text;
  sendMessage();
}
