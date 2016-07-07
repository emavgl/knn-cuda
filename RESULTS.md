K-Nearest Neighbor
===================
Test results
-------------------
##### Orazio Contarino and Emanuele Viglianisi
___

Execution time and comparison
---

#### [INPUT 1] ./generator $((1024*1024)) 2000 50 50

|                            | GTX 755M   | GTX 540M   |
|--------------------------  |----------  |----------- |
| knn_cpu                    | 392366ms   | 442173ms   |
| knn_punisher_v0            | 110434ms   | 362847ms   |
| knn_punisher_v1            | 110333ms   | 362833ms   |
| knn_punisher_self          | 110332ms   | 362828ms   |
| knn_punisher_self          | 110524ms   | 362834ms   |
| knn_punisher_roll_opt      | 20343.1ms  | 96010.5ms  |


#### [INPUT 2] ./generator $((1024*256)) 2000 50 50

|                            | GTX 755M   | GTX 540M    |
|--------------------------  |----------  |-----------  |
| knn_cpu                    | 98789.5ms  | 111811ms    |
| knn_punisher_v0            | 27674.7ms  | 90902.3ms   |
| knn_punisher_v1            | 27631.6ms  | 90882ms     |
| knn_punisher_self          | 27662.3ms  | 90876.8ms   |
| knn_punisher_self          | 27637ms    | 90885.3ms   |
| knn_punisher_roll_opt      | 5110.45ms  | 24119.7ms   |

#### [INPUT 3] ./generator $((1024*256)) 1000 50 50

|                            | GTX 755M   | GTX 540M   |
|--------------------------  |----------  |----------- |
| knn_cpu                    | 48901.7    | 55533.1ms  |
| knn_punisher_v0            | 13721.3ms  | 45413.4ms  |
| knn_punisher_v1            | 113718.4ms | 45412.6ms  |
| knn_punisher_self          | 13718.6ms  | 45412.4ms  |
| knn_punisher_self          | 13717.5ms  | 45412.9ms  |
| knn_punisher_roll_opt      | 2553.26ms  | 12144.8ms  |

#### [INPUT 4] ./generator $((1024*256)) 2000 100 50

|                            | GTX 755M   | GTX 540M    |
|--------------------------  |----------  |-----------  |
| knn_cpu                    | 98018.8ms  | 113326ms    |
| knn_punisher_v0            | 27678.8ms  | 91034.4ms   |
| knn_punisher_v1            | 27679ms    | 91014.5ms   |
| knn_punisher_self          | 27636.7ms  | 91011.7ms   |
| knn_punisher_self          | 27639.4ms  | 91027.7ms   |
| knn_punisher_roll_opt      | 5118.45ms  | 24122.5ms   |

#### [INPUT 5] ./generator $((1024*256)) 2000 50 100

|                            | GTX 755M   | GTX 540M   |
|--------------------------  |----------  |----------- |
| knn_cpu                    | 179835ms   | 207267ms   |
| knn_punisher_v0            | 57435.7ms  | 185390ms   |
| knn_punisher_v1            | 57435.3ms  | 185360ms   |
| knn_punisher_self          | 57407.1ms  | 185383ms   |
| knn_punisher_self          | 54777.5ms  | 185385ms   |
| knn_punisher_roll_opt      | 15514ms    | 44847.5ms  |
___
NVIDIA GTX 755M
---

#### [INPUT 1] ./generator $((1024*1024)) 2000 50 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.94%  138.354s         5  27.6707s  27.6695s  27.6719s  knn
  0.05%  69.799ms        10  6.9799ms  165.86us  13.970ms  [CUDA memcpy HtoD]
  0.01%  19.906ms     10000  1.9900us  1.8550us  71.296us  knnPunisher
  0.00%  14.208us         5  2.8410us  2.7200us  3.0080us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.393s         5  27.6786s  27.6723s  27.6826s  knn
  0.05%  70.291ms        10  7.0291ms  165.76us  14.548ms  [CUDA memcpy HtoD]
  0.00%  1.0096ms         5  201.91us  197.63us  203.26us  knnPunisher
  0.00%  22.176us        10  2.2170us  1.6320us  2.7520us  [CUDA memcpy DtoH]
  0.00%  12.480us         5  2.4960us  2.3680us  2.6880us  [CUDA memset]
  0.00%  10.048us         5  2.0090us  1.9200us  2.2080us  findIntInArray

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.173s         5  27.6346s  27.6323s  27.6363s  knn
  0.05%  70.005ms        10  7.0005ms  165.82us  13.896ms  [CUDA memcpy HtoD]
  0.00%  603.55us        10  60.354us  50.016us  70.655us  knnPunisher
  0.00%  27.776us        15  1.8510us  1.2160us  2.6880us  [CUDA memcpy DtoH]
  0.00%  21.856us        10  2.1850us  2.1120us  2.6880us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.178s         5  27.6356s  27.6332s  27.6387s  knn
  0.05%  69.449ms        10  6.9449ms  165.82us  13.820ms  [CUDA memcpy HtoD]
  0.00%  2.6709ms         5  534.17us  533.60us  534.68us  knnPunisher
  0.00%  14.208us         5  2.8410us  2.8160us  2.8480us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.65%  25.5881s         5  5.11761s  5.11712s  5.11792s  knn
  0.33%  84.202ms        12  7.0168ms  165.41us  14.227ms  [CUDA memcpy HtoD]
  0.02%  4.4684ms         5  893.67us  893.18us  894.27us  knnPunisher
  0.00%  13.664us         5  2.7320us  2.7200us  2.7520us  [CUDA memcpy DtoH]
```

#### [INPUT 2] ./generator $((1024*256)) 2000 50 50
```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.93%  138.338s         5  27.6676s  27.6657s  27.6691s  knn
  0.05%  71.387ms        10  7.1387ms  165.95us  15.412ms  [CUDA memcpy HtoD]
  0.01%  19.651ms     10000  1.9650us  1.8240us  44.608us  knnPunisher
  0.00%  14.112us         5  2.8220us  2.7200us  2.9760us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.180s         5  27.6361s  27.6311s  27.6408s  knn
  0.05%  69.808ms        10  6.9808ms  165.82us  13.880ms  [CUDA memcpy HtoD]
  0.00%  961.85us        10  96.185us  56.896us  132.70us  knnPunisher
  0.00%  28.447us        15  1.8960us  1.4710us  2.4960us  [CUDA memcpy DtoH]
  0.00%  21.855us        10  2.1850us  1.9840us  2.5600us  [CUDA memset]
  0.00%  19.104us        10  1.9100us  1.8240us  2.2080us  findIntInArray

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.309s         5  27.6619s  27.6606s  27.6627s  knn
  0.05%  69.883ms        10  6.9883ms  165.86us  13.855ms  [CUDA memcpy HtoD]
  0.00%  357.21us        10  35.721us  26.879us  44.896us  knnPunisher
  0.00%  27.263us        15  1.8170us  1.2480us  2.3680us  [CUDA memcpy DtoH]
  0.00%  21.760us        10  2.1760us  2.1120us  2.6880us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.174s         5  27.6348s  27.6336s  27.6363s  knn
  0.05%  69.947ms        10  6.9947ms  165.82us  13.864ms  [CUDA memcpy HtoD]
  0.00%  2.2804ms         5  456.09us  455.32us  456.70us  knnPunisher
  0.00%  13.600us         5  2.7200us  2.6880us  2.7520us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.66%  25.5512s         5  5.11023s  5.10955s  5.11206s  knn
  0.33%  83.776ms        12  6.9813ms  165.44us  13.876ms  [CUDA memcpy HtoD]
  0.01%  3.6352ms         5  727.04us  725.98us  728.44us  knnPunisher
  0.00%  13.376us         5  2.6750us  2.6560us  2.7200us  [CUDA memcpy DtoH]
```

#### [INPUT 3] ./generator $((1024*256)) 1000 50 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.89%  68.5881s         5  13.7176s  13.7169s  13.7184s  knn
  0.10%  69.279ms        10  6.9279ms  165.22us  13.735ms  [CUDA memcpy HtoD]
  0.01%  7.1487ms      5000  1.4290us  1.3440us  40.607us  knnPunisher
  0.00%  10.400us         5  2.0800us  2.0160us  2.2720us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.90%  68.5884s         5  13.7177s  13.7168s  13.7185s  knn
  0.10%  69.097ms        10  6.9097ms  165.18us  13.689ms  [CUDA memcpy HtoD]
  0.00%  248.32us         5  49.663us  49.472us  50.048us  knnPunisher
  0.00%  18.016us        10  1.8010us  1.6320us  2.1120us  [CUDA memcpy DtoH]
  0.00%  11.808us         5  2.3610us  2.2400us  2.4000us  [CUDA memset]
  0.00%  6.8800us         5  1.3760us  1.2800us  1.6000us  findIntInArray

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.90%  68.5899s         5  13.7180s  13.7175s  13.7185s  knn
  0.10%  70.167ms        10  7.0167ms  165.18us  13.908ms  [CUDA memcpy HtoD]
  0.00%  199.36us         5  39.872us  39.680us  40.064us  knnPunisher
  0.00%  17.696us        10  1.7690us  1.6000us  2.0160us  [CUDA memcpy DtoH]
  0.00%  11.008us         5  2.2010us  2.1120us  2.5280us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.90%  68.5831s         5  13.7166s  13.7158s  13.7173s  knn
  0.10%  68.959ms        10  6.8959ms  165.18us  13.648ms  [CUDA memcpy HtoD]
  0.00%  853.89us         5  170.78us  170.24us  171.17us  knnPunisher
  0.00%  10.272us         5  2.0540us  2.0480us  2.0800us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.35%  12.7554s         5  2.55108s  2.55031s  2.55305s  knn
  0.64%  82.530ms        12  6.8775ms  165.18us  13.680ms  [CUDA memcpy HtoD]
  0.01%  910.14us         5  182.03us  180.38us  183.20us  knnPunisher
  0.00%  10.240us         5  2.0480us  2.0160us  2.0800us  [CUDA memcpy DtoH]
```

#### [INPUT 4] ./generator $((1024*256)) 2000 100 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.94%  138.354s         5  27.6707s  27.6695s  27.6719s  knn
  0.05%  69.799ms        10  6.9799ms  165.86us  13.970ms  [CUDA memcpy HtoD]
  0.01%  19.906ms     10000  1.9900us  1.8550us  71.296us  knnPunisher
  0.00%  14.208us         5  2.8410us  2.7200us  3.0080us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
  Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.393s         5  27.6786s  27.6723s  27.6826s  knn
  0.05%  70.291ms        10  7.0291ms  165.76us  14.548ms  [CUDA memcpy HtoD]
  0.00%  1.0096ms         5  201.91us  197.63us  203.26us  knnPunisher
  0.00%  22.176us        10  2.2170us  1.6320us  2.7520us  [CUDA memcpy DtoH]
  0.00%  12.480us         5  2.4960us  2.3680us  2.6880us  [CUDA memset]
  0.00%  10.048us         5  2.0090us  1.9200us  2.2080us  findIntInArray

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.173s         5  27.6346s  27.6323s  27.6363s  knn
  0.05%  70.005ms        10  7.0005ms  165.82us  13.896ms  [CUDA memcpy HtoD]
  0.00%  603.55us        10  60.354us  50.016us  70.655us  knnPunisher
  0.00%  27.776us        15  1.8510us  1.2160us  2.6880us  [CUDA memcpy DtoH]
  0.00%  21.856us        10  2.1850us  2.1120us  2.6880us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  138.178s         5  27.6356s  27.6332s  27.6387s  knn
  0.05%  69.449ms        10  6.9449ms  165.82us  13.820ms  [CUDA memcpy HtoD]
  0.00%  2.6709ms         5  534.17us  533.60us  534.68us  knnPunisher
  0.00%  14.208us         5  2.8410us  2.8160us  2.8480us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.65%  25.5881s         5  5.11761s  5.11712s  5.11792s  knn
  0.33%  84.202ms        12  7.0168ms  165.41us  14.227ms  [CUDA memcpy HtoD]
  0.02%  4.4684ms         5  893.67us  893.18us  894.27us  knnPunisher
  0.00%  13.664us         5  2.7320us  2.7200us  2.7520us  [CUDA memcpy DtoH]
```

#### [INPUT 5] ./generator $((1024*256)) 2000 50 100
```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  287.169s         5  57.4337s  57.4218s  57.4411s  knn
  0.05%  137.73ms        10  13.773ms  165.86us  27.497ms  [CUDA memcpy HtoD]
  0.01%  19.421ms     10000  1.9420us  1.8240us  43.488us  knnPunisher
  0.00%  14.240us         5  2.8480us  2.7200us  3.0080us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  287.225s         5  57.4449s  57.4347s  57.4543s  knn
  0.05%  136.10ms        10  13.610ms  165.76us  27.104ms  [CUDA memcpy HtoD]
  0.00%  691.87us         5  138.37us  138.21us  138.50us  knnPunisher
  0.00%  20.288us        10  2.0280us  1.6000us  2.7200us  [CUDA memcpy DtoH]
  0.00%  12.160us         5  2.4320us  2.3680us  2.5280us  [CUDA memset]
  0.00%  10.016us         5  2.0030us  1.9520us  2.2080us  findIntInArray

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  287.005s         5  57.4010s  57.3852s  57.4065s  knn
  0.05%  136.19ms        10  13.619ms  165.79us  27.112ms  [CUDA memcpy HtoD]
  0.00%  215.20us         5  43.039us  42.816us  43.391us  knnPunisher
  0.00%  22.496us        10  2.2490us  1.6000us  2.9120us  [CUDA memcpy DtoH]
  0.00%  11.008us         5  2.2010us  2.0800us  2.5600us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  273.804s         5  54.7608s  54.7500s  54.7768s  knn
  0.05%  137.63ms        10  13.763ms  168.06us  27.440ms  [CUDA memcpy HtoD]
  0.00%  2.4170ms         5  483.39us  483.04us  483.58us  knnPunisher
  0.00%  13.760us         5  2.7520us  2.7200us  2.7840us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.79%  77.5702s         5  15.5140s  15.5109s  15.5178s  knn
  0.20%  158.18ms        12  13.182ms  165.25us  26.844ms  [CUDA memcpy HtoD]
  0.01%  3.9037ms         5  780.74us  776.83us  784.54us  knnPunisher
  0.00%  13.664us         5  2.7320us  2.7200us  2.7520us  [CUDA memcpy DtoH]

```


***
NVIDIA GTX 540M
---

#### [INPUT 1] ./generator $((1024*1024)) 2000 50 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%    2e+03s         5  364.799s  364.794s  364.804s  knn
  0.01%  188.71ms        10  18.871ms  702.07us  37.870ms  [CUDA memcpy HtoD]
  0.00%  83.590ms     10000  8.3580us  8.0820us  226.99us  knnPunisher
  0.00%  14.752us         5  2.9500us  2.5600us  3.2000us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%    2e+03s         5  365.058s  365.049s  365.068s  knn
  0.01%  184.20ms        10  18.420ms  679.51us  36.929ms  [CUDA memcpy HtoD]
  0.00%  1.4538ms         5  290.76us  290.55us  291.18us  knnPunisher
  0.00%  31.167us         5  6.2330us  6.1930us  6.2870us  findIntInArray
  0.00%  22.304us        10  2.2300us  1.6640us  3.2000us  [CUDA memcpy DtoH]
  0.00%  11.872us         5  2.3740us  2.3680us  2.4000us  [CUDA memset]

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%    2e+03s         5  364.962s  364.946s  364.968s  knn
  0.01%  178.42ms        10  17.842ms  678.62us  35.199ms  [CUDA memcpy HtoD]
  0.00%  1.1307ms         5  226.14us  225.38us  226.79us  knnPunisher
  0.00%  23.712us        10  2.3710us  1.6960us  3.4240us  [CUDA memcpy DtoH]
  0.00%  11.647us         5  2.3290us  2.3030us  2.3360us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%    2e+03s         5  365.084s  365.076s  365.093s  knn
  0.01%  185.57ms        10  18.557ms  695.51us  36.599ms  [CUDA memcpy HtoD]
  0.00%  10.251ms         5  2.0502ms  2.0478ms  2.0536ms  knnPunisher
  0.00%  21.536us         5  4.3070us  2.8160us  8.9920us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.95%  490.289s         5  98.0579s  98.0476s  98.0631s  knn
  0.05%  222.66ms        12  18.555ms  695.10us  36.575ms  [CUDA memcpy HtoD]
  0.00%  17.820ms         5  3.5641ms  3.5607ms  3.5664ms  knnPunisher
  0.00%  14.078us         5  2.8150us  2.5590us  3.2000us  [CUDA memcpy DtoH]

```

#### [INPUT 2] ./generator $((1024*256)) 2000 50 50
```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.97%  454.403s         5  90.8807s  90.8757s  90.8874s  knn
  0.02%  83.666ms     10000  8.3660us  8.0360us  254.30us  knnPunisher
  0.01%  48.425ms        10  4.8425ms  164.41us  9.6009ms  [CUDA memcpy HtoD]
  0.00%  13.823us         5  2.7640us  2.5590us  3.2000us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
 99.99%  454.412s         5  90.8824s  90.8780s  90.8900s  knn
  0.01%  46.907ms        10  4.6907ms  163.65us  9.5232ms  [CUDA memcpy HtoD]
  0.00%  2.6905ms        10  269.05us  128.97us  409.44us  knnPunisher
  0.00%  60.934us        10  6.0930us  5.5440us  6.7490us  findIntInArray
  0.00%  29.568us        15  1.9710us  1.4720us  2.5600us  [CUDA memcpy DtoH]
  0.00%  20.512us        10  2.0510us  1.6320us  2.6560us  [CUDA memset]

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  454.388s         5  90.8776s  90.8719s  90.8815s  knn
  0.01%  46.078ms        10  4.6078ms  163.68us  9.1150ms  [CUDA memcpy HtoD]
  0.00%  1.6988ms        10  169.87us  86.732us  253.51us  knnPunisher
  0.00%  497.24us        10  49.724us  2.3360us  162.24us  [CUDA memset]
  0.00%  29.375us        15  1.9580us  1.4400us  2.5920us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  454.401s         5  90.8803s  90.8763s  90.8840s  knn
  0.01%  47.562ms        10  4.7562ms  164.45us  9.4064ms  [CUDA memcpy HtoD]
  0.00%  12.614ms         5  2.5228ms  2.5189ms  2.5261ms  knnPunisher
  0.00%  14.816us         5  2.9630us  2.8160us  3.4560us  [CUDA memcpy DtoH]


=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.94%  120.593s         5  24.1186s  24.1157s  24.1207s  knn(
  0.05%  57.207ms        12  4.7673ms  164.16us  9.4791ms  [CUDA memcpy HtoD]
  0.02%  18.237ms         5  3.6474ms  3.6425ms  3.6519ms  knnPunisher
  0.00%  14.688us         5  2.9370us  2.7840us  3.4240us  [CUDA memcpy DtoH]

```

#### [INPUT 3] ./generator $((1024*256)) 1000 50 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.97%  227.051s         5  45.4101s  45.4080s  45.4123s  knn
  0.02%  44.764ms        10  4.4764ms  163.10us  9.1066ms  [CUDA memcpy HtoD]
  0.01%  18.187ms      5000  3.6370us  3.4290us  121.79us  knnPunisher
  0.00%  10.496us         5  2.0990us  1.9840us  2.5280us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.98%  227.047s         5  45.4093s  45.4073s  45.4118s  knn
  0.02%  44.636ms        10  4.4635ms  163.10us  8.8776ms  [CUDA memcpy HtoD]
  0.00%  716.71us         5  143.34us  143.06us  143.70us  knnPunisher
  0.00%  19.392us        10  1.9390us  1.6960us  2.3680us  [CUDA memcpy DtoH]
  0.00%  19.057us         5  3.8110us  3.8030us  3.8240us  findIntInArray
  0.00%  12.224us         5  2.4440us  2.3680us  2.6560us  [CUDA memset]

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.98%  227.046s         5  45.4093s  45.4033s  45.4116s  knn
  0.02%  46.057ms        10  4.6057ms  163.10us  9.1551ms  [CUDA memcpy HtoD]
  0.00%  649.30us         5  129.86us  2.4000us  172.54us  [CUDA memset]
  0.00%  606.81us         5  121.36us  120.98us  121.93us  knnPunisher
  0.00%  20.672us        10  2.0670us  1.6960us  2.6560us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.98%  227.048s         5  45.4097s  45.4054s  45.4116s  knn
  0.02%  44.359ms        10  4.4359ms  163.07us  8.7944ms  [CUDA memcpy HtoD]
  0.00%  3.5580ms         5  711.59us  709.09us  714.27us  knnPunisher
  0.00%  11.680us         5  2.3360us  2.2400us  2.6560us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.90%  60.7171s         5  12.1434s  12.1430s  12.1440s  knn
  0.09%  55.561ms        12  4.6301ms  162.88us  9.2242ms  [CUDA memcpy HtoD]
  0.01%  4.8014ms         5  960.29us  959.32us  960.84us  knnPunisher
  0.00%  11.168us         5  2.2330us  2.1440us  2.5600us  [CUDA memcpy DtoH]
```

#### [INPUT 4] ./generator $((1024*256)) 2000 100 50

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.97%  455.050s         5  91.0099s  91.0032s  91.0135s  knn
  0.02%  84.471ms     10000  8.4470us  8.0380us  363.57us  knnPunisher
  0.01%  48.657ms        10  4.8657ms  164.48us  9.8841ms  [CUDA memcpy HtoD]
  0.00%  13.504us         5  2.7000us  2.5600us  3.2000us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  455.061s         5  91.0121s  91.0086s  91.0180s  knn
  0.01%  46.236ms        10  4.6236ms  163.77us  9.1394ms  [CUDA memcpy HtoD]
  0.00%  3.5902ms         5  718.03us  715.99us  719.75us  knnPunisher
  0.00%  29.152us         5  5.8300us  5.8190us  5.8540us  findIntInArray
  0.00%  22.272us        10  2.2270us  1.6960us  3.1680us  [CUDA memcpy DtoH]
  0.00%  12.032us         5  2.4060us  2.3360us  2.6560us  [CUDA memset]

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  455.054s         5  91.0107s  91.0078s  91.0122s  knn
  0.01%  47.495ms        10  4.7495ms  163.65us  9.4700ms  [CUDA memcpy HtoD]
  0.00%  2.5214ms        10  252.13us  141.27us  363.45us  knnPunisher
  0.00%  29.727us        15  1.9810us  1.4400us  2.5920us  [CUDA memcpy DtoH]
  0.00%  29.216us        10  2.9210us  2.2720us  7.7440us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  455.056s         5  91.0112s  90.9993s  91.0170s  knn
  0.01%  47.455ms        10  4.7455ms  163.68us  9.4222ms  [CUDA memcpy HtoD]
  0.00%  15.484ms         5  3.0969ms  3.0952ms  3.0982ms  knnPunisher
  0.00%  13.312us         5  2.6620us  2.5280us  3.1360us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.94%  120.561s         5  24.1121s  24.1078s  24.1159s  knn
  0.05%  56.833ms        12  4.7361ms  163.65us  9.6674ms  [CUDA memcpy HtoD]
  0.02%  18.986ms         5  3.7971ms  3.7956ms  3.8000ms  knnPunisher
  0.00%  14.720us         5  2.9440us  2.8160us  3.4560us  [CUDA memcpy DtoH]

```

#### [INPUT 5] ./generator $((1024*256)) 2000 50 100

```
=== KNN_PUNISHER_V0 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.98%  926.885s         5  185.377s  185.367s  185.386s  knn
  0.01%  95.442ms        10  9.5442ms  163.65us  19.327ms  [CUDA memcpy HtoD]
  0.01%  83.250ms     10000  8.3240us  8.0380us  247.76us  knnPunisher
  0.00%  13.472us         5  2.6940us  2.5600us  3.2000us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_V1 ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  926.906s         5  185.381s  185.357s  185.401s  knn
  0.01%  95.672ms        10  9.5672ms  163.71us  19.208ms  [CUDA memcpy HtoD]
  0.00%  2.1185ms         5  423.71us  423.14us  424.18us  knnPunisher
  0.00%  32.265us         5  6.4530us  6.4150us  6.4930us  findIntInArray
  0.00%  22.400us        10  2.2400us  1.6960us  3.1680us  [CUDA memcpy DtoH]
  0.00%  12.000us         5  2.4000us  2.3360us  2.6560us  [CUDA memset]

=== KNN_PUNISHER_V1_SELF ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  926.818s         5  185.364s  185.357s  185.380s  knn
  0.01%  95.125ms        10  9.5125ms  164.51us  19.038ms  [CUDA memcpy HtoD]
  0.00%  1.2360ms         5  247.21us  246.70us  247.81us  knnPunisher
  0.00%  23.744us        10  2.3740us  1.6640us  3.4880us  [CUDA memcpy DtoH]
  0.00%  11.680us         5  2.3360us  2.2720us  2.5920us  [CUDA memset]

=== KNN_PUNISHER_ROLL ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.99%  926.872s         5  185.374s  185.361s  185.381s  knn
  0.01%  93.832ms        10  9.3832ms  163.68us  19.071ms  [CUDA memcpy HtoD]
  0.00%  10.313ms         5  2.0626ms  2.0608ms  2.0655ms  knnPunisher
  0.00%  14.976us         5  2.9950us  2.8480us  3.4880us  [CUDA memcpy DtoH]

=== KNN_PUNISHER_ROLL_OPT ===
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.94%  224.243s         5  44.8486s  44.8429s  44.8567s  knn
  0.05%  115.23ms        12  9.6021ms  164.41us  19.099ms  [CUDA memcpy HtoD]
  0.01%  17.711ms         5  3.5421ms  3.5378ms  3.5475ms  knnPunisher
  0.00%  14.752us         5  2.9500us  2.8160us  3.4560us  [CUDA memcpy DtoH]

```
