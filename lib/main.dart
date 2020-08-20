import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite/tflite.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(new App());

const String ssd = "SSD MobileNet";
const String NumDetector = "SSD MobileNet Num";
const String GaugeDetector = "SSD MobileNet Gauge";
const String GaugeNum = "GaugeNum";

class App extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyApp(),
    );
  }
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => new _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File _image;
  List _recognitions;
  String _model = GaugeNum;
  double _imageHeight;
  double _imageWidth;
  bool _busy = false;

  Future predictImagePicker() async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    setState(() {
      _busy = true;
    });
    predictImage(image);
  }

  Future predictImage(File image) async {
    if (image == null) return;

    switch (_model) {
      case ssd:
        await ssdMobileNet(image);
        break;
      case NumDetector:
        await ssdMobileNet(image);
        break;
      case GaugeDetector:
        await ssdMobileNet(image);
        break;
      case GaugeNum:
        image = await GNDetector(image);
        break;
    }

    new FileImage(image)
        .resolve(new ImageConfiguration())
        .addListener(ImageStreamListener((ImageInfo info, bool _) {
      setState(() {
        _imageHeight = info.image.height.toDouble();
        _imageWidth = info.image.width.toDouble();
      });
    }));

    setState(() {
      _image = image;
      _busy = false;
    });
  }

  @override
  void initState() {
    super.initState();

    _busy = true;

    loadModel().then((val) {
      setState(() {
        _busy = false;
      });
    });
  }

  Future loadModel() async {
    Tflite.close();
    try {
      String res;
      switch (_model) {
        case NumDetector:
          res = await Tflite.loadModel(
            model: "assets/Num.tflite",
            labels: "assets/Numlabel.txt",
            // useGpuDelegate: true,
          );
          break;
        case GaugeDetector:
          res = await Tflite.loadModel(
            model: "assets/GaugeDetector.tflite",
            labels: "assets/Gaugelabel.txt",
            // useGpuDelegate: true,
          );
          break;
        case GaugeNum:
          break;
      }
      print(res);
    } on PlatformException {
      print('Failed to load model.');
    }
  }

  Uint8List imageToByteListFloat32(
      img.Image image, int inputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Uint8List imageToByteListUint8(img.Image image, int inputSize) {
    var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
    var buffer = Uint8List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = img.getRed(pixel);
        buffer[pixelIndex++] = img.getGreen(pixel);
        buffer[pixelIndex++] = img.getBlue(pixel);
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Future ssdMobileNet(File image) async {
    int startTime = new DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      numResultsPerClass: 1,
      threshold: 0.5,
    );
//     var imageBytes = (await rootBundle.load(image.path)).buffer;
//     img.Image oriImage = img.decodeJpg(imageBytes.asUint8List());
//     img.Image resizedImage = img.copyResize(oriImage, width: 300, height: 300);
//     var recognitions = await Tflite.detectObjectOnBinary(
//       binary: imageToByteListUint8(resizedImage, 300),
//       numResultsPerClass: 1,
//     );
    setState(() {
      _recognitions = recognitions;
    });
    int endTime = new DateTime.now().millisecondsSinceEpoch;
    print("Inference took ${endTime - startTime}ms");
  }

  Future<File> GNDetector(File image) async {
    int startTime = new DateTime.now().millisecondsSinceEpoch;
    imageCache.clear();
    var res = await Tflite.loadModel(
      model: "assets/GaugeDetector.tflite",
      labels: "assets/Gaugelabel.txt",
      // useGpuDelegate: true,
    );
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      numResultsPerClass: 1,
      threshold: 0.5,
    );
    Tflite.close();
    //load Num Model
    res = await Tflite.loadModel(
      model: "assets/Num.tflite",
      labels: "assets/Numlabel.txt",
      // useGpuDelegate: true,
    );
    Directory tempDir = await getTemporaryDirectory();
    String tempPath = tempDir.path;
    var imageBytes = image.readAsBytesSync().buffer;
    img.Image oriImage = img.decodeJpg(imageBytes.asUint8List());
    int oriImgW = oriImage.width;
    int oriImgH = oriImage.height;
    int x = (oriImage.width * recognitions[0]['rect']['x']).toInt();
    int w = (oriImage.width * recognitions[0]['rect']['w']).toInt();
    int y = (oriImage.height * recognitions[0]['rect']['y']).toInt();
    int h = (oriImage.height * recognitions[0]['rect']['h']).toInt();
    img.Image cropImage = img.copyCrop(oriImage, x, y, w, h);
    int cropImgW = cropImage.width;
    int cropImgH = cropImage.height;
//    File('$tempPath/cropTmp_$startTime.jpg').writeAsBytesSync(img.encodeJpg(cropImage));
//    File cropImgF = File('$tempPath/cropTmp.jpg');
    double xScale = 0.3;
    double yScale = 1;
    //partition Right and Left
    //------------------Right-------------//
    int PartRightx = (cropImgW*xScale).toInt();
    int PartRighty = 0;
    int PartRightw = cropImgW - PartRightx;
    int PartRighth = (cropImgH*yScale).toInt();
    print("x:"+PartRightx.toString()+"y:"+PartRighty.toString()+"w:"+PartRightw.toString()+"h:"+PartRighth.toString());
    img.Image PartImgLeft = img.copyCrop(cropImage, PartRightx, PartRighty, PartRightw, PartRighth);
    int PartImgLeftW = PartImgLeft.width;
    int PartImgLeftH = PartImgLeft.height;
//    File('$tempPath/PartLeftTmp_$startTime.jpg').writeAsBytesSync(img.encodeJpg(PartImgLeft));
//    File pasteImgF = File('$tempPath/PartLeftTmp_$startTime.jpg');

    int pastWidth = PartImgLeftW;
    int pastHeight = PartImgLeftW - PartImgLeftH;
    img.Image pasteImg = img.Image(pastWidth, pastHeight).fill(0);
    int pastX = 0;
    int pastY = (pastHeight / 2).toInt() - (PartImgLeftH / 2).toInt();
    int pastYMax = (pastHeight / 2).toInt() + (PartImgLeftH / 2).toInt();
    print(pastHeight);
    print(cropImgH);
    print(pastY);
    for(int i = 0; i < pasteImg.width; i++){
      for(int j = pastY; j < pastYMax; j++){
        pasteImg.setPixel(i, j, PartImgLeft.getPixel(i, j-pastY));
        //print(pasteImg.getPixel(i, j));
      }
    }
    File('$tempPath/PasteLeftTmp_$startTime.jpg').writeAsBytesSync(img.encodeJpg(pasteImg));
    File pasteImgF = File('$tempPath/PasteLeftTmp_$startTime.jpg');



    if(res == "success"){
      recognitions = await Tflite.detectObjectOnImage(
          path: pasteImgF.path,
          numResultsPerClass: 1,
          threshold: 0.1,
      );
    }
    print(recognitions);
//     var imageBytes = (await rootBundle.load(image.path)).buffer;
//     img.Image oriImage = img.decodeJpg(imageBytes.asUint8List());
//     img.Image resizedImage = img.copyResize(oriImage, width: 300, height: 300);
//     var recognitions = await Tflite.detectObjectOnBinary(
//       binary: imageToByteListUint8(resizedImage, 300),
//       numResultsPerClass: 1,
//     );
    setState(() {
      _recognitions = recognitions;
    });
    int endTime = new DateTime.now().millisecondsSinceEpoch;
    print("Inference took ${endTime - startTime}ms");
    return pasteImgF;
  }

  onSelect(model) async {
    setState(() {
      _busy = true;
      _model = model;
      _recognitions = null;
    });
    await loadModel();

    if (_image != null)
      predictImage(_image);
    else
      setState(() {
        _busy = false;
      });
  }

  List<Widget> renderBoxes(Size screen) {
    if (_recognitions == null) return [];
    if (_imageHeight == null || _imageWidth == null) return [];

    double factorX = screen.width;
    double factorY = _imageHeight / _imageWidth * screen.width;
    Color blue = Color.fromRGBO(37, 213, 253, 1.0);
    return _recognitions.map((re) {
      return Positioned(
        left: re["rect"]["x"] * factorX,
        top: re["rect"]["y"] * factorY,
        width: re["rect"]["w"] * factorX,
        height: re["rect"]["h"] * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.all(Radius.circular(8.0)),
            border: Border.all(
              color: blue,
              width: 2,
            ),
          ),
          child: Text(
            "${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%",
            style: TextStyle(
              background: Paint()..color = blue,
              color: Colors.white,
              fontSize: 12.0,
            ),
          ),
        ),
      );
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];
    stackChildren.add(Positioned(
      top: 0.0,
      left: 0.0,
      width: size.width,
      child: _image == null ? Text('No image selected.') : Image.file(_image),
    ));

    if (_model == NumDetector || _model == GaugeDetector || _model == GaugeNum) {
      stackChildren.addAll(renderBoxes(size));
    }

    if (_busy) {
      stackChildren.add(const Opacity(
        child: ModalBarrier(dismissible: false, color: Colors.grey),
        opacity: 0.3,
      ));
      stackChildren.add(const Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('tflite demo'),
        actions: <Widget>[
          PopupMenuButton<String>(
            onSelected: onSelect,
            itemBuilder: (context) {
              List<PopupMenuEntry<String>> menuEntries = [
                const PopupMenuItem<String>(
                  child: Text(NumDetector),
                  value: NumDetector,
                ),
                const PopupMenuItem<String>(
                  child: Text(GaugeDetector),
                  value: GaugeDetector,
                ),
                const PopupMenuItem<String>(
                  child: Text(GaugeNum),
                  value: GaugeNum,
                )
              ];
              return menuEntries;
            },
          )
        ],
      ),
      body: Stack(
        children: stackChildren,
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: predictImagePicker,
        tooltip: 'Pick Image',
        child: Icon(Icons.image),
      ),
    );
  }
}