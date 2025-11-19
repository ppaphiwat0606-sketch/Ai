let model;
const imageInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predict-btn");
const predictionsContainer = document.getElementById("predictions");

// โหลดโมเดล
async function loadModel() {
  try {
    model = await tf.loadLayersModel("model/model.json");
    console.log("✅ โหลดโมเดลสำเร็จ");
  } catch (err) {
    console.error("❌ โหลดโมเดลไม่สำเร็จ:", err);
    alert("โหลดโมเดลไม่สำเร็จ! ตรวจสอบว่า model/model.json อยู่ในที่ถูกต้อง");
  }
}
loadModel();

// เมื่อเลือกภาพ
imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = "block";
    predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
});

// ปุ่มทำนาย
predictBtn.addEventListener("click", async () => {
  if (!model) {
    alert("ยังโหลดโมเดลไม่เสร็จ!");
    return;
  }

  const imgTensor = tf.browser.fromPixels(preview)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();

  const prediction = await model.predict(imgTensor).data();
  showPredictions(prediction);
  tf.dispose(imgTensor);
});

// แสดงผล
function showPredictions(predictions) {
  predictionsContainer.innerHTML = "";

  // 👇 label เกม
  const labels = ["Fortnite", "Minecraft", "Terraria", "Roblox"];

  const results = Array.from(predictions)
    .map((p, i) => ({ label: labels[i] || `Class ${i + 1}`, prob: p }))
    .sort((a, b) => b.prob - a.prob);

  results.forEach((r) => {
    const item = document.createElement("div");
    item.classList.add("prediction-item");

    const percent = (r.prob * 100).toFixed(2);

    item.innerHTML = `
      <strong>${r.label}</strong> — ${percent}%
      <div class="progress-bar" style="width:0%"></div>
    `;

    predictionsContainer.appendChild(item);

    // Animate progress bar
    setTimeout(() => {
      item.querySelector(".progress-bar").style.width = `${percent}%`;
    }, 100);
  });
}
