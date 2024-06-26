# OsteoGA

## Prerequisite
To run the application on local machines, Python 3.11.8 is best suited.
Running in other versions may cause package resolution failures, which can lead to the package not having the necessary imported names.
Therefore, we recommend running the app using a Docker container.

## Usage
To get started, follow these steps:

### 1. Build the Docker Image

Ensure that Docker daemon is running, then execute the following command to build the image:

```bash
docker build -t be .
```

### 2. Run the application
Replace `CHOSEN_PORT` with your desired port number, and execute the following command to run the application:
```bash
docker run -dp CHOSEN_PORT:8000 --name osteoga-core be
```
Now, your application should be up and running on the specified port.

### 3. Stop the application
To stop, use:
```bash
docker stop osteoga-core
```

### 4. Remove the application
First, remove the container:
```bash
docker rm osteoga-core
```
Then the image:
```bash
docker rmi be
```

## API description

### POST /predict

This endpoint is used to process an image and return various processed versions of it along with the classification probabilities.

#### Request

The request should be a JSON object with the following properties:

- `image`: A base64 encoded string of the image to be processed. This is a required field.
- `crop`: A string that indicates whether the image should be cropped or not. This is an optional field. If not provided, it defaults to `"false"`. If provided, it should be either `"true"` or `"false"`.
- `clinical`: A JSON object that contains the clinical data. This is a required field and should have the following properties:
  - `age`: The age of the patient. This is a required field.
  - `max_weight`, `weight`: The maximum weight and current weight (in kg) of the patient. These are required fields.
  - `height`: The height of the patient, measured in centimeters. This is a required field.

Example:

```json
{
    "image": "base64_encoded_image_string",
    "clinical": {
        "age": 70,
        "max_weight": 100,
        "height": 175,
        "weight": 75
    },
    "crop": "true"
}
```

#### Response

The response will be a JSON object with the following properties:

* `images`: A dictionary where the keys are the names of the processed images and the values are base64 encoded strings of the images. The keys can be `cropped`, `segmented`, `contour`, `dilated`, `blurred`, `masked`, `restored`, and `anomaly`.

* `probabilities`: A list of probabilities resulting from the classification of the restored image. If an error occurred during the classification, this will be an empty list.

* `explanation_image_html`: ...

* `error`: A string that describes the error that occurred during the processing of the image. If no error occurred, this will be `null`.

* `results`: A dictionary that contains the classification results. It has the following keys:
    - `kl_grade`: A list of probabilities for the Kellgren-Lawrence (KL) grade classification.
    - `jsn = ['jsn_def', 'jsn_mild/mod', 'jsn_none', 'jsn_severe']`: A list of probabilities for the joint space narrowing (JSN) classification.
    - `osteophytes = ['osteophytes_def','osteophytes_none', 'osteophytes_poss']`: A list of probabilities for the osteophytes classification.
    - For example:
    ```json
        "results": {
            "jsn": [
                0.0330350436270237,
                0.5218521952629089,
                0.1046275943517685,
                0.2277068793773651
            ],
            "kl_grade": [
                0.07987314285243892,
                0.025284186845900128,
                0.07620671808468844,
                0.8086584760492599,
                0.009977476167712234
            ],
            "osteophytes": [
                0.9177196025848389,
                0.03786254674196243,
                0.05698298662900925
            ]
        }
    ```

#### Errors

The endpoint may return one of the following errors:

| Error Code | Description |
|------------|-------------|
| `missing_clinical_data` | This error is returned when the clinical object is not in the request object. |
| `no_object_detected` | This error is returned when no object is detected in the image and cropping is requested. |
| `invalid_input_format` | This error is returned when the input image cannot be processed due to an invalid format. |
| `segmentation_failed` | This error is returned when the segmentation of the image fails. |
| `contour_extraction_failed` | This error is returned when the extraction of the contours from the image fails. |
| `masking_failed` | This error is returned when the masking of the image fails. |
| `restoration_failed` | This error is returned when the restoration of the image fails. |
| `anomaly_map_failed` | This error is returned when the creation of the anomaly map fails. |
| `classification_failed` | This error is returned when the classification of the restored image fails. |

Each error is accompanied by a `500` status code.

### POST /predictall

> **IMPORTANT NOTE:** For this endpoint, cropping is always enabled to ensure all potential knees in the image are scanned.

This endpoint is used to process an image and return various processed versions of it along with the classification probabilities for each detected object in the image.

#### Request

The request should be a JSON object with the following properties:

- `image`: A base64 encoded string of the image to be processed. This is a required field.
- `clinical`: A JSON object that contains the clinical data. This is a required field and should have the following properties:
  - `age`: The age of the patient. This is a required field.
  - `max_weight`, `weight`: The maximum weight and current weight (in kg) of the patient. These are required fields.
  - `height`: The height of the patient, measured in centimeters. This is a required field.

Example:

```json
{
    "image": "base64_encoded_image_string",
    "clinical": {
        "age": 70,
        "max_weight": 100,
        "height": 175,
        "weight": 75
    },
}
```

#### Response

The response will be a JSON object with the following properties:

* `objects`: A list of objects detected in the image. Each object is a dictionary with the following properties:
    * `images`: A dictionary where the keys are the names of the processed images and the values are base64 encoded strings of the images. The keys can be `cropped`, `segmented`, `contour`, `dilated`, `blurred`, `masked`, `restored`, and `anomaly`.
    * `probabilities`: A list of probabilities resulting from the classification of the restored image. If an error occurred during the classification, this will be an empty list.
    * `error`: A string that describes the error that occurred during the processing of the image. If no error occurred, this will be `null`.
    * `results`:results: A dictionary that contains the classification results. It has the following keys:
        - `kl_grade`: A list of probabilities for the Kellgren-Lawrence (KL) grade classification.
        - `jsn = ['jsn_def', 'jsn_mild/mod', 'jsn_none', 'jsn_severe']`: A list of probabilities for the joint space narrowing (JSN) classification.
        - `osteophytes = ['osteophytes_def','osteophytes_none', 'osteophytes_poss']`: A list of probabilities for the osteophytes classification.
    * `explanation_image_html`: ...
    * For example:
    ```json
        "results": {
            "jsn": [
                0.0330350436270237,
                0.5218521952629089,
                0.1046275943517685,
                0.2277068793773651
            ],
            "kl_grade": [
                0.07987314285243892,
                0.025284186845900128,
                0.07620671808468844,
                0.8086584760492599,
                0.009977476167712234
            ],
            "osteophytes": [
                0.9177196025848389,
                0.03786254674196243,
                0.05698298662900925
            ]
        }
    ```

* `length`: The number of objects detected in the image.

* `error`: A string that describes the error that occurred during the processing of the image. If no error occurred, this will be `null`.

#### Errors

The endpoint may return one of the following errors:

| Error Code | Description |
|------------|-------------|
| `missing_clinical_data` | This error is returned when the clinical object is not in the request object. |
| `no_object_detected` | This error is returned when no object is detected in the image. |
| `invalid_input_format` | This error is returned when the input image cannot be processed due to an invalid format. |
| `cropping_failed` | This error is returned when the cropping of the image fails. |
| `segmentation_failed` | This error is returned when the segmentation of the image fails. |
| `contour_extraction_failed` | This error is returned when the extraction of the contours from the image fails. |
| `masking_failed` | This error is returned when the masking of the image fails. |
| `restoration_failed` | This error is returned when the restoration of the image fails. |
| `anomaly_map_failed` | This error is returned when the creation of the anomaly map fails. |
| `classification_failed` | This error is returned when the classification of the restored image fails. |

Each error is accompanied by a `500` status code. If the request is processed successfully, the status code will be `200`.