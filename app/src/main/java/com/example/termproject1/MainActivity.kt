package com.example.termproject1

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import org.tensorflow.lite.Interpreter

class MainActivity : AppCompatActivity(), View.OnClickListener {
    private val TAG = MainActivity::class.java.simpleName
    private var imageView: ImageView? = null
    private var animalImageView: ImageView? = null
    private var startButton: Button? = null
    private var actionButton: Button? = null
    private var exitButton: Button? = null
    private var mCurrentPhotoPath: String? = null
    private var photoURI: Uri? = null
    private var albumURI: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 레이아웃 요소 초기화
        imageView = findViewById(R.id.imageview)
        animalImageView = findViewById(R.id.animal_imageview)
        startButton = findViewById(R.id.start_button)
        actionButton = findViewById(R.id.action_button)
        exitButton = findViewById(R.id.exit_button)

        // 클릭 리스너 설정
        startButton!!.setOnClickListener(this)
        actionButton!!.setOnClickListener(this)
        exitButton!!.setOnClickListener(this)

        // 퍼미션 체크
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Permission: ${permissions[0]} was ${grantResults[0]}")
        }
    }

    // 갤러리에서 이미지 선택 후 결과 처리
    @RequiresApi(api = Build.VERSION_CODES.N)
    override fun onActivityResult(requestCode: Int, resultCode: Int, intent: Intent?) {
        super.onActivityResult(requestCode, resultCode, intent)
        try {
            when (requestCode) {
                REQUEST_TAKE_ALBUM -> if (resultCode == Activity.RESULT_OK) {
                    handleImageFromGallery(intent)
                }

                REQUEST_IMAGE_CROP -> if (resultCode == Activity.RESULT_OK) {
                    galleryAddPic()
                    imageView!!.setImageURI(albumURI)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // 갤러리에서 선택한 이미지 처리
    private fun handleImageFromGallery(intent: Intent?) {
        if (intent?.data != null) {
            try {
                photoURI = intent.data
                val stream: InputStream? = contentResolver.openInputStream(photoURI!!)
                val bitmap = BitmapFactory.decodeStream(stream)
                stream?.close()

                if (bitmap != null) {
                    imageView!!.setImageBitmap(bitmap)
                    processImage(bitmap)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error while handling image from gallery: ${e.message}")
            }
        }
    }

    // 이미지 처리
    private fun processImage(bitmap: Bitmap) {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true)
        val inputArray = normalizeImageToFloatArray(resizedBitmap)
        val interpreter = getTfliteInterpreter("mnist.tflite")
        val outputArray = Array(1) { FloatArray(10) }
        interpreter?.run(inputArray, outputArray)
        displayPredictionResult(outputArray[0])
    }

    // 이미지를 정규화하여 배열로 변환
    private fun normalizeImageToFloatArray(bitmap: Bitmap): Array<Array<FloatArray>> {
        val bytesImg = Array(1) { Array(28) { FloatArray(28) } }
        for (y in 0 until 28) {
            for (x in 0 until 28) {
                val pixel = bitmap.getPixel(x, y)
                val gray = (0.299 * (pixel shr 16 and 0xff) + 0.587 * (pixel shr 8 and 0xff) + 0.114 * (pixel and 0xff)).toInt()
                bytesImg[0][y][x] = gray / 255f
            }
        }
        return bytesImg
    }

    // 예측 결과 표시
    private fun displayPredictionResult(predictions: FloatArray) {
        val tvResult: TextView = findViewById(R.id.result_0)
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: -1
        when (maxIndex) {
            2 -> displayAnimalResult(tvResult, predictions[2], R.drawable.cat_image, "고양이")
            3 -> displayAnimalResult(tvResult, predictions[3], R.drawable.hamster_image, "햄스터")
            5 -> displayAnimalResult(tvResult, predictions[5], R.drawable.dog_image, "강아지")
            else -> tvResult.text = "분석 결과를 찾을 수 없습니다."
        }
        actionButton!!.text = "다시하기"
        exitButton!!.visibility = View.VISIBLE
    }

    // 동물 예측 결과 표시
    private fun displayAnimalResult(textView: TextView, probability: Float, imageResource: Int, animalName: String) {
        val probabilityPercentage = probability * 100
        textView.text = String.format("%.2f%% 확률로 $animalName 상", probabilityPercentage)
        animalImageView?.setImageResource(imageResource)
    }

    // TensorFlow Lite 모델을 읽어오는 함수
    private fun getTfliteInterpreter(modelPath: String): Interpreter? {
        return try {
            Interpreter(loadModelFile(this@MainActivity, modelPath))
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // 모델 파일을 로드하는 함수
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, modelPath: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // 갤러리에서 이미지 선택하는 함수
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Throws(IOException::class)
    private fun album() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = MediaStore.Images.Media.CONTENT_TYPE
        intent.type = "image/*"
        startActivityForResult(intent, REQUEST_TAKE_ALBUM)
    }

    // 갤러리에 이미지 추가하는 함수
    private fun galleryAddPic() {
        val mediaScanIntent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
        val f = File(mCurrentPhotoPath!!)
        val contentUri = Uri.fromFile(f)
        mediaScanIntent.data = contentUri
        sendBroadcast(mediaScanIntent)
    }

    // 클릭 이벤트 처리
    override fun onClick(v: View) {
        when (v.id) {
            R.id.start_button -> showAnalysisScreen()
            R.id.action_button -> {
                if (actionButton!!.text == "갤러리") {
                    album()
                } else {
                    resetScreen()
                }
            }
            R.id.exit_button -> showExitScreen()
        }
    }

    // 분석 화면 보여주기
    private fun showAnalysisScreen() {
        imageView!!.visibility = View.VISIBLE
        animalImageView!!.visibility = View.VISIBLE
        findViewById<LinearLayout>(R.id.textviews_container).visibility = View.VISIBLE
        actionButton!!.visibility = View.VISIBLE
        startButton!!.visibility = View.GONE
        exitButton!!.visibility = View.GONE
    }

    // 화면 초기화
    private fun resetScreen() {
        imageView!!.setImageDrawable(null)
        animalImageView!!.setImageDrawable(null)
        findViewById<TextView>(R.id.result_0).text = ""
        actionButton!!.text = "갤러리"
        exitButton!!.visibility = View.GONE
    }

    // 종료 화면 보여주기
    private fun showExitScreen() {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
    }

    companion object {
        private const val REQUEST_TAKE_ALBUM = 2
        private const val REQUEST_IMAGE_CROP = 3
    }
}
