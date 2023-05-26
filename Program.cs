using Newtonsoft.Json;
using System.Numerics;
using FftSharp;
using MathNet.Numerics.Statistics;
using Complex = System.Numerics.Complex;
using System;

namespace SETIDecode;

public class MetaData
{
    public Dictionary<string, string> global { get; set; }
    public List<Capture> captures { get; set; }
}

public class Capture
{
    [JsonProperty("core:datetime")]
    public string DateTime { get; set; }
    [JsonProperty("core:frequency")]
    public double Frequency { get; set; }
    [JsonProperty("core:sample_start")]
    public int SampleStart { get; set; }
}

public class Program
{

    // Size of data chunk to process at a time (bytes)
    private static readonly int ChunkSize = 268435456;  // With 128GB RAM, 536870912 works for the first stage (conversion to floats) but not the second (merge, which OOMs). 134217728 is too small given the start time difference between the files.
    private static Dictionary<string, Dictionary<string, object>> allFileMetrics;

    public static void Main()
    {
        #region Metrics and float files
        // Check if metrics file exists
        var metricsFile = "allFileMetrics.json";
        if (File.Exists(metricsFile))
        {
            // Load allFileMetrics from file
            var json = File.ReadAllText(metricsFile);
            allFileMetrics = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, object>>>(json);
            Console.WriteLine("Float file conversions already complete and metrics loaded from file.");
        }
        else
        {
            Console.WriteLine("Gathering metrics and converting original files to float data...");

            // Initialize allFileMetrics
            allFileMetrics = new Dictionary<string, Dictionary<string, object>>();

            // List of all .sigmf-meta files
            var metaFiles = new List<string>
            {
                "A_Sign_in_Space-ATA-X.sigmf-meta",
                "A_Sign_in_Space-ATA-Y.sigmf-meta",
                "A_Sign_in_Space-GBT.sigmf-meta",
                "A_Sign_in_Space-Medicina.sigmf-meta"
            };

            foreach (var metaFile in metaFiles)
            {
                // Load .sigmf-meta file
                var metaData = JsonConvert.DeserializeObject<MetaData>(File.ReadAllText(metaFile));
                var fileMetrics = new Dictionary<string, object>();

                // Extract core:datetime and core:frequency
                var capture = metaData.captures.First();
                var dateTime = DateTime.Parse(capture.DateTime);
                var frequency = capture.Frequency;

                Console.WriteLine($"{metaFile}: dateTime = {dateTime}, frequency = {frequency}");

                // Load corresponding .sigmf-data file
                var dataFile = metaFile.Replace(".sigmf-meta", ".sigmf-data");
                using var stream = new FileStream(dataFile, FileMode.Open, FileAccess.Read);
                using var reader = new BinaryReader(stream);
                var floatDataFile = metaFile.Replace(".sigmf-meta", "_float.sigmf-data");
                using var floatStream = new FileStream(floatDataFile, FileMode.Create, FileAccess.Write);
                using var floatWriter = new BinaryWriter(floatStream);
                var chunkMaxMagnitudes = new List<double>();
                var chunkMeans = new List<Complex>();
                var chunkPowers = new List<double>();
                var chunkVariances = new List<double>();
                var chunkSNRs = new List<double>();
                var numBuckets = 75; // If this proves too many, 35 is a decent number (there's some obvious bleed from the buckets that have higher power into adjacent buckets though)
                List<double[]> chunkPowerBuckets = new List<double[]>();

                // Read ci8 IQ data in chunks and convert to Complex
                while (reader.BaseStream.Position != reader.BaseStream.Length)
                {
                    var buffer = reader.ReadBytes(ChunkSize * 2);
                    var complexData = new List<Complex>();
                    for (var i = 0; i < buffer.Length; i += 2)
                    {
                        var real = buffer[i] / 128.0;
                        var imaginary = buffer[i + 1] / 128.0;
                        var complex = new Complex(real, imaginary);

                        complexData.Add(complex);

                        floatWriter.Write((float)real);
                        floatWriter.Write((float)imaginary);
                    }

                    // Compute mean and maxMagnitude for this chunk
                    var chunkMeanReal = complexData.Average(c => c.Real);
                    var chunkMeanImaginary = complexData.Average(c => c.Imaginary);
                    var chunkMean = new Complex(chunkMeanReal, chunkMeanImaginary);
                    var chunkMaxMagnitude = complexData.Max(c => c.Magnitude);

                    // Prepare signal for FFT
                    var signal = complexData.Select(c => c.Real).ToArray();
                    var paddedSignal = new double[NextPowerOfTwo(signal.Length)];
                    Array.Copy(signal, paddedSignal, signal.Length);

                    // Apply window and compute FFT
                    var window = new FftSharp.Windows.Hanning();
                    window.ApplyInPlace(paddedSignal);
                    var spectrum = FFT.Forward(paddedSignal);

                    // Compute power spectral density and divide into buckets
                    var power = spectrum.Select(s => Math.Pow(s.Magnitude, 2) / 100000).ToArray();
                    double[] powerBuckets = new double[numBuckets];
                    for (int i = 0; i < power.Length; i++)
                    {
                        int bucket = (int)(i * (double)((double)numBuckets / power.Length));
                        powerBuckets[bucket] += power[i];
                    }
                    for (int i = 0; i < numBuckets; i++)
                    {
                        powerBuckets[i] /= power.Length / numBuckets;
                    }
                    chunkPowerBuckets.Add(powerBuckets);

                    // Compute power mean, variance, and SNR for this chunk
                    var chunkPower = power.Average();
                    var chunkVariance = Statistics.Variance(signal) + 1e-10;  // add small constant to avoid division by zero
                    var chunkSNR = 10 * Math.Log10((chunkPower * 100000) / chunkVariance);  // calculate SNR in decibels

                    //Console.WriteLine($"\t{dataFile} (chunk ending at {reader.BaseStream.Position}): mean = {chunkMean}, maxMagnitude = {chunkMaxMagnitude}, power = {chunkPower}, variance = {chunkVariance}, SNR = {chunkSNR}, powerBuckets = [{string.Join(", ", powerBuckets.ToList().ConvertAll(x => (int)Math.Round(x)))}]");

                    // Store chunk values for later computation
                    chunkMaxMagnitudes.Add(chunkMaxMagnitude);
                    chunkMeans.Add(chunkMean);
                    chunkPowers.Add(chunkPower);
                    chunkVariances.Add(chunkVariance);
                    chunkSNRs.Add(chunkSNR);
                }

                floatWriter.Close();
                floatStream.Close();

                // Compute overall mean and maxMagnitude
                var overallMeanReal = chunkMeans.Average(c => c.Real);
                var overallMeanImaginary = chunkMeans.Average(c => c.Imaginary);
                var overallMean = new Complex(overallMeanReal, overallMeanImaginary);
                var overallMaxMagnitude = chunkMaxMagnitudes.Max();
                double[] overallPowerBuckets = new double[numBuckets];
                for (int i = 0; i < numBuckets; i++)
                {
                    overallPowerBuckets[i] = chunkPowerBuckets.Average(b => b[i]);
                }

                fileMetrics["dateTime"] = dateTime;
                fileMetrics["frequency"] = frequency;
                fileMetrics["overallPowerBuckets"] = overallPowerBuckets;
                fileMetrics["overallMean"] = overallMean;
                fileMetrics["overallMaxMagnitude"] = overallMaxMagnitude;
                fileMetrics["averagePower"] = chunkPowers.Average();
                fileMetrics["averageSNR"] = chunkSNRs.Average();
                allFileMetrics[floatDataFile] = fileMetrics;

                Console.WriteLine($"{floatDataFile} (overall): mean = {overallMean}, maxMagnitude = {overallMaxMagnitude}, power = {chunkPowers.Average()}, variance = {chunkVariances.Average()}, SNR = {chunkSNRs.Average()}, powerBuckets = [{string.Join(", ", overallPowerBuckets.ToList().ConvertAll(x => (int)Math.Round(x)))}]\n");
            }

            // Save allFileMetrics to file
            var json = JsonConvert.SerializeObject(allFileMetrics);
            File.WriteAllText(metricsFile, json);
        }

        string mergedMetaFile = "A_Sign_in_Space_Merged.sigmf-meta";
        var mergedMetaData = new MetaData
        {
            global = new Dictionary<string, string>(),
            captures = new List<Capture>()
        };
        if (File.Exists(mergedMetaFile))
        {
            // Load allFileMetrics from file
            var json = File.ReadAllText(mergedMetaFile);
            mergedMetaData = JsonConvert.DeserializeObject<MetaData>(json);
            Console.WriteLine("Merged data and metadata files already exist; merged metadata loaded from file.");
        }
        else
        {
            Console.WriteLine("Merging data to a single float file and metadata file...");

            // List of all *_float.sigmf-data files
            var floatFiles = new List<string>
            {
                "A_Sign_in_Space-ATA-X_float.sigmf-data",
                "A_Sign_in_Space-ATA-Y_float.sigmf-data",
                "A_Sign_in_Space-GBT_float.sigmf-data",
                "A_Sign_in_Space-Medicina_float.sigmf-data"
            };

            #endregion Metrics and float files

            #region Align and merge
            // Select the observatory with the earliest timestamp to be the reference.
            string referenceFile = allFileMetrics.OrderBy(x => x.Value["dateTime"]).First().Key;
            double referenceFrequency = (double)allFileMetrics[referenceFile]["frequency"];

            // Prepare BinaryReader instances for the data streams.
            Dictionary<string, BinaryReader> readers = new Dictionary<string, BinaryReader>();
            foreach (var floatFile in floatFiles)
            {
                var stream = new FileStream(floatFile, FileMode.Open, FileAccess.Read);
                readers[floatFile] = new BinaryReader(stream);
            }
            // Set the capture metadata to the start of each data stream.
            foreach (var floatFile in floatFiles)
            {
                mergedMetaData.captures.Add(new Capture
                {
                    DateTime = allFileMetrics[floatFile]["dateTime"].ToString(),
                    Frequency = (double)allFileMetrics[floatFile]["frequency"],
                    SampleStart = 0 // TODO: Not sure this is what we want...
                });
            }

            // Prepare a FileStream and BinaryWriter for the merged data.
            string mergedDataFile = "A_Sign_in_Space_Merged.sigmf-data";
            using var mergedStream = new FileStream(mergedDataFile, FileMode.Create, FileAccess.Write);
            using var mergedWriter = new BinaryWriter(mergedStream);

            int chunkCt = 0;
            // Process the data in chunks.
            while (true)
            {
                // Load the next chunk from each data stream.
                Console.WriteLine($"\tLoading chunk from files...");
                // Prepare buffers for the chunk data streams.
                Dictionary<string, List<Complex>> buffers = new Dictionary<string, List<Complex>>();
                foreach (var floatFile in floatFiles)
                {
                    buffers[floatFile] = new List<Complex>();
                }
                int chunkLength = ChunkSize;
                // Load the chunk from each file.
                foreach (var floatFile in floatFiles)
                {
                    for (int i = 0; i < ChunkSize; i++)
                    {
                        if (readers[floatFile].BaseStream.Position != readers[floatFile].BaseStream.Length)
                        {
                            var real = readers[floatFile].ReadSingle();
                            var imaginary = readers[floatFile].ReadSingle();
                            buffers[floatFile].Add(new Complex(real, imaginary));
                        }
                        else
                        {
                            // If the end of the file has been reached, pad with zeros.
                            buffers[floatFile].Add(new Complex(0, 0));

                            // Adjust the chunk length if necessary.
                            if (i < chunkLength)
                            {
                                chunkLength = i;
                            }
                        }
                    }
                }

                if (chunkLength == 0)
                {
                    // If all files have been fully read, break the loop.
                    break;
                }
                chunkCt++;
                Console.WriteLine($"\tProcessing chunk #{chunkCt}...");

                // Process the frequency shift.
                Console.WriteLine("\t\tShifting frequencies...");
                foreach (var floatFile in floatFiles)
                {
                    if (floatFile != referenceFile)
                    {
                        double frequencyDifference = referenceFrequency - (double)allFileMetrics[floatFile]["frequency"];
                        if (Math.Abs(frequencyDifference) > 0)
                        {
                            double shiftPhasePerSample = 2.0 * Math.PI * frequencyDifference / 1000000.0;
                            double shiftPhase = 0;
                            for (int i = 0; i < buffers[floatFile].Count; i++)
                            {
                                Complex shiftFactor = Complex.FromPolarCoordinates(1, shiftPhase);
                                buffers[floatFile][i] *= shiftFactor;
                                shiftPhase += shiftPhasePerSample;
                            }
                        }
                    }
                }

                // Align and normalize the data streams.
                int maxDataLength = buffers.Max(x => x.Value.Count);
                DateTime referenceTime = (DateTime)allFileMetrics[referenceFile]["dateTime"];
                // Coarse alignment.
                Console.WriteLine("\t\tCoarse alignment...");
                foreach (var floatFile in floatFiles)
                {
                    // Get the time difference in seconds.
                    double timeDifference = ((DateTime)allFileMetrics[floatFile]["dateTime"] - referenceTime).TotalSeconds;

                    // Convert the time difference to the number of samples.
                    int sampleDifference = (int)(timeDifference * 1000000);

                    // Shift the data.
                    if (sampleDifference > 0)
                    {
                        // If the file started after the reference, remove the first 'sampleDifference' samples.
                        buffers[floatFile].RemoveRange(0, sampleDifference);
                    }
                    else
                    {
                        // If the file started before the reference, prepend with zeros.
                        buffers[floatFile].InsertRange(0, new Complex[-sampleDifference]);
                    }

                    // Align the lengths.
                    int alignmentZeros = maxDataLength - buffers[floatFile].Count;
                    buffers[floatFile].InsertRange(0, new Complex[alignmentZeros]);
                }
                // TODO: Fine alignment
                Console.WriteLine($"\t\tFINE ALIGNMENT NOT IMPLEMENTED!");
                // Some rough/pseudo code for it...
                /*var bucketsToMatch = new[] { 1, 14, 62, 75 }; // The buckets where we will look for peaks.
                var paddedBuffer = new double[NextPowerOfTwo(buffers[referenceFile].Count)];
                Array.Copy(buffers[referenceFile], paddedBuffer, buffers[referenceFile].Count);
                // Apply window and compute FFT
                var window = new FftSharp.Windows.Hanning();
                window.ApplyInPlace(paddedBuffer);
                var referenceFftData = FFT.Forward(paddedBuffer);
                var referencePeaks = FindPeaks(referenceFftData, bucketsToMatch); // Replace with your function to find peaks.
                foreach (var floatFile in floatFiles)
                {
                    if (floatFile == referenceFile) continue;

                    paddedBuffer = new double[NextPowerOfTwo(buffers[floatFile].Count)];
                    Array.Copy(buffers[floatFile], paddedBuffer, buffers[floatFile].Count);
                    // Apply window and compute FFT
                    window = new FftSharp.Windows.Hanning();
                    window.ApplyInPlace(paddedBuffer);
                    var candidateFftData = FFT.Forward(paddedBuffer);

                    var candidatePeaks = FindPeaks(candidateFftData, bucketsToMatch); // Find peaks.

                    // Find the shift that gives the best match between the peaks.
                    var shift = FindBestShift(referencePeaks, candidatePeaks); // Replace with your function to find the best shift.

                    // Shift the data.
                    if (shift > 0)
                    {
                        buffers[floatFile].RemoveRange(0, shift);
                    }
                    else
                    {
                        buffers[floatFile].InsertRange(0, new Complex[-shift]);
                    }

                    // Normalize data by dividing by the max magnitude.
                    double maxMagnitude = buffers[floatFile].Max(x => x.Magnitude);
                    for (int i = 0; i < buffers[floatFile].Count; i++)
                    {
                        buffers[floatFile][i] /= maxMagnitude;
                    }
                }*/
                // Normalization to [-1,1]
                Console.WriteLine("\t\tNormalization...");
                foreach (var floatFile in floatFiles)
                {
                    // Normalize data by dividing by the max magnitude.
                    double maxMagnitude = buffers[floatFile].Max(x => x.Magnitude);
                    for (int i = 0; i < buffers[floatFile].Count; i++)
                    {
                        buffers[floatFile][i] /= maxMagnitude;
                    }
                }

                // Merge the data streams by performing an intelligent average.
                Console.WriteLine("\t\tAveraging (for each sample, average the two ATA values, then from the three values find two closest in magnitude and average those)...");
                List <Complex> mergedData = new List<Complex>(maxDataLength);
                for (int i = 0; i < maxDataLength; i++)
                {
                    Complex ataAverage = new Complex(0, 0);
                    Complex gbtValue = new Complex(0, 0);
                    Complex medValue = new Complex(0, 0);
                    int ataCount = 0;
                    int gbtCount = 0;
                    int medCount = 0;

                    if (buffers.ContainsKey("A_Sign_in_Space-ATA-X_float.sigmf-data") && i < buffers["A_Sign_in_Space-ATA-X_float.sigmf-data"].Count)
                    {
                        ataAverage += buffers["A_Sign_in_Space-ATA-X_float.sigmf-data"][i];
                        ataCount++;
                    }

                    if (buffers.ContainsKey("A_Sign_in_Space-ATA-Y_float.sigmf-data") && i < buffers["A_Sign_in_Space-ATA-Y_float.sigmf-data"].Count)
                    {
                        ataAverage += buffers["A_Sign_in_Space-ATA-Y_float.sigmf-data"][i];
                        ataCount++;
                    }

                    if (buffers.ContainsKey("A_Sign_in_Space-GBT_float.sigmf-data") && i < buffers["A_Sign_in_Space-GBT_float.sigmf-data"].Count)
                    {
                        gbtValue = buffers["A_Sign_in_Space-GBT_float.sigmf-data"][i];
                        gbtCount++;
                    }

                    if (buffers.ContainsKey("A_Sign_in_Space-Medicina_float.sigmf-data") && i < buffers["A_Sign_in_Space-Medicina_float.sigmf-data"].Count)
                    {
                        medValue = buffers["A_Sign_in_Space-Medicina_float.sigmf-data"][i];
                        medCount++;
                    }

                    if (ataCount > 0) ataAverage /= ataCount;

                    // Find the two values that are most similar in magnitude.
                    double ataMagnitude = ataAverage.Magnitude;
                    double gbtMagnitude = gbtValue.Magnitude;
                    double medMagnitude = medValue.Magnitude;

                    // Calculate the absolute differences in magnitude.
                    double ataGbtDiff = Math.Abs(ataMagnitude - gbtMagnitude);
                    double ataMedDiff = Math.Abs(ataMagnitude - medMagnitude);
                    double gbtMedDiff = Math.Abs(gbtMagnitude - medMagnitude);

                    // Initialize the sum and count for the final averaging.
                    Complex sum = new Complex(0, 0);
                    int count = 0;

                    if (ataGbtDiff <= ataMedDiff && ataGbtDiff <= gbtMedDiff)
                    {
                        // ATA and GBT are most similar.
                        if (ataCount > 0)
                        {
                            sum += ataAverage;
                            count++;
                        }
                        if (gbtCount > 0)
                        {
                            sum += gbtValue;
                            count++;
                        }
                    }
                    else if (ataMedDiff <= ataGbtDiff && ataMedDiff <= gbtMedDiff)
                    {
                        // ATA and Medicina are most similar.
                        if (ataCount > 0)
                        {
                            sum += ataAverage;
                            count++;
                        }
                        if (medCount > 0)
                        {
                            sum += medValue;
                            count++;
                        }
                    }
                    else
                    {
                        // GBT and Medicina are most similar.
                        if (gbtCount > 0)
                        {
                            sum += gbtValue;
                            count++;
                        }
                        if (medCount > 0)
                        {
                            sum += medValue;
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        // Calculate the final average and add it to the merged data.
                        mergedData.Add(sum / count);
                    }
                    else
                    {
                        // If no data streams are available at this index, add a zero complex number.
                        mergedData.Add(new Complex(0, 0));
                    }
                }


                // Temporary nonsense just to sanity check the data (which currently fails, presumably because of no fine alignment)
                Console.WriteLine($"\t\tCalculating merged chunk FFT PSD...");
                int numBuckets = 75;
                var paddedBuffer = new double[NextPowerOfTwo(mergedData.Count)];
                Array.Copy(mergedData.Select(c => c.Real).ToArray(), paddedBuffer, mergedData.Count);
                // Apply window and compute FFT
                var window = new FftSharp.Windows.Hanning();
                window.ApplyInPlace(paddedBuffer);
                var referenceFftData = FFT.Forward(paddedBuffer);
                // Compute power spectral density and divide into buckets
                var power = referenceFftData.Select(s => Math.Pow(s.Magnitude, 2) / 7100).ToArray();
                double[] powerBuckets = new double[numBuckets];
                for (int i = 0; i < power.Length; i++)
                {
                    int bucket = (int)(i * (double)((double)numBuckets / power.Length));
                    powerBuckets[bucket] += power[i];
                }
                for (int i = 0; i < numBuckets; i++)
                {
                    powerBuckets[i] /= power.Length / numBuckets;
                }
                Console.WriteLine($"\t\tMerged chunk FFT PSD buckets: [{string.Join(", ", powerBuckets.ToList().ConvertAll(x => (int)Math.Round(x)))}]");


                // Write the merged data to the output file.
                Console.WriteLine($"\t\tWriting merged data to output file...");
                for (int i = 0; i < chunkLength; i++)
                {
                    mergedWriter.Write((float)mergedData[i].Real);
                    mergedWriter.Write((float)mergedData[i].Imaginary);
                }
            }

            // Close the BinaryReader instances.
            foreach (var reader in readers.Values)
            {
                reader.Close();
            }

            // Write the metadata for the merged data to a new .sigmf-meta file...

            // Set the global metadata to the earliest datetime and the reference frequency.
            mergedMetaData.global["core:datetime"] = allFileMetrics[referenceFile]["dateTime"].ToString();
            mergedMetaData.global["core:frequency"] = referenceFrequency.ToString();

            // Serialize and write the metadata to the .sigmf-meta file.
            var jsonOut = JsonConvert.SerializeObject(mergedMetaData);
            File.WriteAllText(mergedMetaFile, jsonOut);

            Console.WriteLine("Data merging completed. Merged data and metadata files have been created.");
        }
        #endregion Align and merge

        Console.ReadLine();
    }

    public static int NextPowerOfTwo(int x)
    {
        if (x == 0)
        {
            return 1;
        }

        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }
}