using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using System.Collections.Generic;
using System.Linq;



namespace CMP304_Iris_Classifier
{


    public class InputData
    {
        [LoadColumn(0)]
        public float SepalLength { get; set; }

        [LoadColumn(1)]
        public float SepalWidth { get; set; }

        [LoadColumn(2)]
        public float PetalLength { get; set; }

        [LoadColumn(3)]
        public float PetalWidth { get; set; }

        [LoadColumn(4)]
        public string Species { get; set; }
    }


    public class InputDataImages
    {
        [LoadColumn(0)]
        public string Label { get; set; }

        [LoadColumn(1)]
        public float LeftEyebrow { get; set; }

        [LoadColumn(2)]
        public float RightEyebrow { get; set; }

        [LoadColumn(3)]
        public float LeftLip { get; set; }

        [LoadColumn(4)]
        public float RightLip { get; set; }

        [LoadColumn(5)]
        public float LipWidth { get; set; }
        [LoadColumn(6)]
        public float LipHeight { get; set; }


    }


    public class SpeciesPrediction
    {
        [ColumnName("PredictedLabel")]
        //Describes the Species predicted by the ML
        public string Species { get; set; }

        [ColumnName("Score")]
        // Degree of confidence for each category 
        public float[] Scores { get; set; }


    }


    class Program
    {
        static void Main(string[] args)
        {

            var mlContext = new MLContext();

            // Need to create a DataView (FOR IMAGES)
            IDataView dataView = mlContext.Data.LoadFromTextFile<InputDataImages>("feature_vectors.csv",
                hasHeader: true, separatorChar: ',');


            // Defining the models pipeline
            var FeatureVectorName = "Features";
            var LabelColumnName = "Label";

            var pipeline = mlContext.Transforms.
                Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: LabelColumnName)
                .Append(mlContext.Transforms.Concatenate(FeatureVectorName, "LeftEyebrow", "RightEyebrow", "LeftLip", "RightLip", "LipWidth", "LipHeight"))
                .AppendCacheCheckpoint(mlContext).
                Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(LabelColumnName, FeatureVectorName))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(dataView);

            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            { mlContext.Model.Save(model, dataView.Schema, fileStream); }


            //Define Predictor
            var predictor = mlContext.Model.CreatePredictionEngine<InputDataImages, SpeciesPrediction>(model);


            // Create list to store paths for images 
            List<string> Paths = Directory.GetFiles("Neutral", "*.png").ToList();
            //string[] NeutralImagesPaths = Directory.GetFiles("TestingSetNeutral", "*.png").ToArray();
            Paths.AddRange(Directory.GetFiles("Fear", "*.png").ToList());
            Paths.AddRange(Directory.GetFiles("Disgust", "*.png").ToList());
            Paths.AddRange(Directory.GetFiles("Anger", "*.png").ToList());


            // Create header for CSV file 
            string Header = "Label, Score1, Score2, Score3, Score4, Path \n";
            string HeaderImages = "Label, LeftEyebrow, RightEyebrow, LeftLip, RightLip, LipWidth, LipHeight \n";
            System.IO.File.WriteAllText(@"TestingResults.csv", Header);
            System.IO.File.WriteAllText(@"TestingFeatureVectorValues.csv", HeaderImages);

            // Load Image and compute feature values
            //string str = "S116_006_02240709.png";

            foreach (var str in Paths)
            {
                InputDataImages inputValues = GetFeaturesValuesFromImage(str);
                //compute label from parent's directory name

                //var prediction = predictor.Predict(new InputDataImages()
                //{
                //    LeftEyebrow = inputValues.LeftEyebrow,
                //    RightEyebrow = inputValues.RightEyebrow,
                //    LeftLip = inputValues.LeftLip,
                //    RightLip = inputValues.RightLip,
                //    LipWidth = inputValues.LipWidth,
                //    LipHeight = inputValues.LipHeight

                //});

                //Console.Write($"*** Prediction: {prediction.Species} ***");
                //Console.Write($"*** Scores: {string.Join(" ", prediction.Scores)} ***");
                //Console.Write($"*** Path: {str} ***");
                //Console.WriteLine();


            }

            List<InputDataImages> TestingFeatureVectorValues = ReadCSVFile("TestingFeatureVectorValues.csv");


            foreach (var Input in TestingFeatureVectorValues)
            {

                var prediction = predictor.Predict(new InputDataImages()
                {
                    LeftEyebrow = Input.LeftEyebrow,
                    RightEyebrow = Input.RightEyebrow,
                    LeftLip = Input.LeftLip,
                    RightLip = Input.RightLip,
                    LipWidth = Input.LipWidth,
                    LipHeight = Input.LipHeight

                });


                //Write resutls inside CSV file
                using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"TestingResults.csv", true))
                {
                    file.WriteLine(prediction.Species.ToString() + "," + prediction.Scores[0].ToString()
                        + "," + prediction.Scores[1].ToString() + "," + prediction.Scores[2].ToString() + "," +
                        prediction.Scores[3].ToString() + "," + Input.Label.ToString());
                }




            }
            //InputDataImages inputValues = GetFeaturesValuesFromImage(str);
            // SeinputFilePathst up Dlib Face Detector

            //var prediction = predictor.Predict(new InputDataImages()
            //{
            //    LeftEyebrow = inputValues.LeftEyebrow,
            //    RightEyebrow = inputValues.RightEyebrow,
            //    LeftLip = inputValues.LeftLip,
            //    RightLip = inputValues.RightLip,
            //    LipWidth = inputValues.LipWidth,
            //    LipHeight = inputValues.LipHeight

            //});




            // THIS IS JUST FOR GETTING THE MICROACCURACY


            var TestDataView = mlContext.Data.LoadFromTextFile<InputDataImages>("TestingFeatureVectorValues.csv", hasHeader: true, separatorChar: ',');
            var TestMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(dataView));

            Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data"); 
            Console.WriteLine($"* MicroAccuracy:    {TestMetrics.MicroAccuracy:0.###}"); 
            Console.WriteLine($"* MacroAccuracy:    {TestMetrics.MacroAccuracy:0.###}"); 
            Console.WriteLine($"* LogLoss:          {TestMetrics.LogLoss:#.###}"); 
            Console.WriteLine($"* LogLossReduction: {TestMetrics.LogLossReduction:#.###}");

            //Console.Write($"*** Prediction: {prediction.Species} ***");
            //Console.Write($"*** Scores: {string.Join(" ", prediction.Scores)} ***");



        }

            static List<InputDataImages> ReadCSVFile(string path)
        {
            using (var reader = new StreamReader(@path))
            {
                List<InputDataImages> listA = new List<InputDataImages>();
                //List<string> listB = new List<string>();
                reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    InputDataImages inputClass = new InputDataImages();

                    inputClass.Label = values[0].ToString();
                    inputClass.LeftEyebrow = float.Parse(values[1], System.Globalization.CultureInfo.InvariantCulture);
                    inputClass.RightEyebrow = float.Parse(values[2], System.Globalization.CultureInfo.InvariantCulture);
                    inputClass.LeftLip = float.Parse(values[3], System.Globalization.CultureInfo.InvariantCulture);
                    inputClass.RightLip = float.Parse(values[4], System.Globalization.CultureInfo.InvariantCulture);
                    inputClass.LipWidth = float.Parse(values[5], System.Globalization.CultureInfo.InvariantCulture);
                    inputClass.LipHeight = float.Parse(values[6], System.Globalization.CultureInfo.InvariantCulture);

                    listA.Add(inputClass);

                }

                return listA;
            }
        }



        private static InputDataImages GetFeaturesValuesFromImage(string str)
        {
            var returnClass = new InputDataImages();

            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape DetectorS
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {


                // load input image
                var img = Dlib.LoadImage<RgbPixel>(str);

                // find all faces i n the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks


                // Create the CSV file and fill in the first line with the header
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // draw the landmark points on the image
                    for (var i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                    }

                    /////////////// WEEK 9 LAB ////////////////

                    double[] LeftEyebrowDistances = new double[4];
                    double[] RightEyebrowDistances = new double[4];

                    float LeftEyebrowSum = 0;
                    float RightEyebrowSum = 0;

                    //LIP VARIABLES
                    double[] LeftLipDistances = new double[4];
                    double[] RightLipDistances = new double[4];
                    float LeftLipSum = 0;
                    float RightLipSum = 0;


                    LeftEyebrowDistances[0] = (shape.GetPart(21) - shape.GetPart(39)).Length;
                    LeftEyebrowDistances[1] = (shape.GetPart(20) - shape.GetPart(39)).Length;
                    LeftEyebrowDistances[2] = (shape.GetPart(19) - shape.GetPart(39)).Length;
                    LeftEyebrowDistances[3] = (shape.GetPart(18) - shape.GetPart(39)).Length;

                    RightEyebrowDistances[0] = (shape.GetPart(22) - shape.GetPart(42)).Length;
                    RightEyebrowDistances[1] = (shape.GetPart(23) - shape.GetPart(42)).Length;
                    RightEyebrowDistances[2] = (shape.GetPart(24) - shape.GetPart(42)).Length;
                    RightEyebrowDistances[3] = (shape.GetPart(25) - shape.GetPart(42)).Length;


                    //LIP
                    LeftLipDistances[0] = (shape.GetPart(51) - shape.GetPart(33)).Length;
                    LeftLipDistances[1] = (shape.GetPart(50) - shape.GetPart(33)).Length;
                    LeftLipDistances[2] = (shape.GetPart(49) - shape.GetPart(33)).Length;
                    LeftLipDistances[3] = (shape.GetPart(48) - shape.GetPart(33)).Length;


                    RightLipDistances[0] = (shape.GetPart(51) - shape.GetPart(33)).Length;
                    RightLipDistances[1] = (shape.GetPart(52) - shape.GetPart(33)).Length;
                    RightLipDistances[2] = (shape.GetPart(53) - shape.GetPart(33)).Length;
                    RightLipDistances[3] = (shape.GetPart(54) - shape.GetPart(33)).Length;


                    for (int i = 0; i < 4; i++)
                    {
                        LeftEyebrowSum += (float)(LeftEyebrowDistances[i] / LeftEyebrowDistances[0]);
                        RightEyebrowSum += (float)(RightEyebrowDistances[i] / RightEyebrowDistances[0]);

                    }

                    LeftLipSum += (float)(LeftLipDistances[1] / LeftLipDistances[0]);
                    LeftLipSum += (float)(LeftLipDistances[2] / LeftLipDistances[0]);
                    LeftLipSum += (float)(LeftLipDistances[3] / LeftLipDistances[0]);


                    RightLipSum += (float)(RightLipDistances[1] / RightLipDistances[0]);
                    RightLipSum += (float)(RightLipDistances[2] / RightLipDistances[0]);
                    RightLipSum += (float)(RightLipDistances[3] / RightLipDistances[0]);

                    double LipWidth = (float)((shape.GetPart(48) - shape.GetPart(54)).Length / (shape.GetPart(33) - shape.GetPart(51)).Length);
                    double LipHeight = (float)((shape.GetPart(51) - shape.GetPart(57)).Length / (shape.GetPart(33) - shape.GetPart(51)).Length);

                    returnClass.LeftEyebrow = LeftEyebrowSum;
                    returnClass.RightEyebrow = RightLipSum;
                    returnClass.LeftLip = LeftLipSum;
                    returnClass.RightLip = RightLipSum;
                    returnClass.LipWidth = (float)LipWidth;
                    returnClass.LipHeight = (float)LipHeight;


                    // export the modified image
                    string filePath = "output" + ".jpg";
                    Dlib.SaveJpeg(img, filePath);
                }
            }

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"TestingFeatureVectorValues.csv", true))
            {
                DirectoryInfo dr = new DirectoryInfo(str);
                //Console.WriteLine(dr.Parent.Name.ToString());
                string ParentFolderName = dr.Parent.Name.ToString();

                file.WriteLine(ParentFolderName + "," + returnClass.LeftEyebrow.ToString() + "," + returnClass.RightEyebrow.ToString()
                    + "," + returnClass.LeftLip.ToString() + "," + returnClass.RightLip.ToString() + "," + returnClass.LipWidth.ToString()
                    + "," + returnClass.LipHeight.ToString());
            }
            return returnClass;

        }

    }


}
