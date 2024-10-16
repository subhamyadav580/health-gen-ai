// Function to preview the uploaded image
function previewImage() {
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");

    const file = imageInput.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block"; // Show the image preview
    };

    if (file) {
        reader.readAsDataURL(file); // Read the uploaded file as a Data URL
    }
}

// Function to send user input (text or image) to the backend
function sendMessage() {
    const userInput = document.getElementById("userInput").value.trim();
    const patientHistory = document.getElementById("patientHistory").value.trim();
    const imageInput = document.getElementById("imageInput");
    const loader = document.getElementById("loader");


    // Show the loader
    loader.style.display = "block";

    // If an image is selected, send it to the image endpoint
    if (imageInput.files.length > 0) {
        const formData = new FormData();
        formData.append("file", imageInput.files[0]);
        formData.append("patient_history", patientHistory);

        // Display the user image message in the chat
        const imagePreview = document.getElementById("imagePreview");
        appendMessage("user-message", `<img src="${imagePreview.src}" alt="Uploaded Image" style="max-width: 100%; margin-top: 10px;">`);


        fetch("/diagnose_image", {
            method: "POST",
            body: formData
        })
            .then(handleResponse)
            .catch(handleError);

    // If no image is selected, send the text input to the text endpoint
    } else if (userInput) {
        appendMessage("user-message", userInput); // Show user message

        fetch("/diagnose_text", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ report_data: userInput, patient_history: patientHistory })
        })
            .then(handleResponse)
            .catch(handleError);
    } else {
        // Hide the loader and alert if no input is given
        loader.style.display = "none";
        alert("Please enter symptoms or select an image to upload.");
    }

    // Clear the input fields after sending
    document.getElementById("userInput").value = '';
    imageInput.value = ''; // Optionally clear the image input
    document.getElementById("imagePreview").style.display = "none"; // Hide the image preview
}

// Function to handle API responses
function handleResponse(response) {
    return response.json().then(data => {
        const loader = document.getElementById("loader");
        loader.style.display = "none"; // Hide the loader

        // Format the response to replace new lines with <br> tags
        const formattedResponse = formatResponse(data.response);
        // Display the bot's response
        appendMessage("bot-message", formattedResponse);
    });
}

// Function to handle errors
function handleError(error) {
    const loader = document.getElementById("loader");
    loader.style.display = "none"; // Hide the loader
    console.error("Error:", error);
}

// Function to append messages to the chat
function appendMessage(className, message) {
    const chat = document.getElementById("chat");
    const newMessage = document.createElement("div");
    newMessage.className = "chat-bubble " + className;
    newMessage.innerHTML = message; // Use innerHTML to render HTML
    chat.appendChild(newMessage);
}

// Function to format the response for display
function formatResponse(response) {
    // Replace new lines with <br> tags for line breaks and format bold text
    return response.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}
