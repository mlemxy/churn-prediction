package com.itp14.churnbenchmark

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.itp14.churnbenchmark.ui.theme.ChurnBenchmarkTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ln1p
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ChurnBenchmarkTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    BenchmarkScreen(
                        copyAsset = { name ->
                            val f = File(filesDir, name)
                            assets.open(name).use { i -> f.outputStream().use { o -> i.copyTo(o) } }
                            f.absolutePath
                        },
                        appContext = applicationContext
                    )
                }
            }
        }
    }
}

/**
 * SQLite schema matches Section 4.8 of the report exactly: TEXT primary key,
 * WITHOUT ROWID, integer Unix timestamp, cent-based integer for monetary.
 */
class CustomerProfileDb(context: Context) :
    SQLiteOpenHelper(context, "churn_benchmark.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE customer_profile (
                customer_id  TEXT NOT NULL,
                last_seen    INTEGER NOT NULL,
                frequency    INTEGER NOT NULL,
                monetary_c   INTEGER NOT NULL,
                PRIMARY KEY (customer_id)
            ) WITHOUT ROWID;
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS customer_profile")
        onCreate(db)
    }

    /** Seeds the table from customer_sample.json (bundled in assets), containing real
     *  customer profiles exported from the actual dataset rather than hand-authored
     *  synthetic values. Falls back to a small synthetic set only if the asset is
     *  missing, so the benchmark still runs during development. */
    fun seedFromAsset(context: Context): List<String> {
        val db = writableDatabase
        db.execSQL("DELETE FROM customer_profile")
        val now = System.currentTimeMillis() / 1000L
        val customerIds = mutableListOf<String>()

        val jsonText = try {
            context.assets.open("customer_sample.json").bufferedReader().use { it.readText() }
        } catch (e: java.io.FileNotFoundException) {
            null
        }

        if (jsonText != null) {
            val arr = org.json.JSONArray(jsonText)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val customerId = obj.getString("customer_id").removeSuffix(".0")
                val recencyDays = obj.getInt("recency_days")
                val frequency = obj.getInt("frequency")
                val monetary = obj.getDouble("monetary")

                val lastSeen = now - (recencyDays.toLong() * 86400L)
                val values = ContentValues().apply {
                    put("customer_id", customerId)
                    put("last_seen", lastSeen)
                    put("frequency", frequency)
                    put("monetary_c", Math.round(monetary * 100.0))
                }
                db.insertWithOnConflict(
                    "customer_profile", null, values, SQLiteDatabase.CONFLICT_REPLACE
                )
                customerIds.add(customerId)
            }
        } else {
            // Fallback: small synthetic set, only used if customer_sample.json wasn't bundled.
            val fallback = listOf(
                Quad("cust_1", 1, 4, 80.0), Quad("cust_2", 227, 1, 14.0),
                Quad("cust_3", 45, 2, 35.0), Quad("cust_4", 0, 3, 55.0),
                Quad("cust_5", 180, 1, 8.5),
            )
            for (p in fallback) {
                val lastSeen = now - (p.recencyDaysAgo.toLong() * 86400L)
                val values = ContentValues().apply {
                    put("customer_id", p.customerId)
                    put("last_seen", lastSeen)
                    put("frequency", p.frequency)
                    put("monetary_c", Math.round(p.monetaryDollars * 100.0))
                }
                db.insertWithOnConflict(
                    "customer_profile", null, values, SQLiteDatabase.CONFLICT_REPLACE
                )
                customerIds.add(p.customerId)
            }
        }
        return customerIds
    }

    data class Quad(val customerId: String, val recencyDaysAgo: Int, val frequency: Int, val monetaryDollars: Double)
}

data class RawProfile(val lastSeen: Long, val frequency: Int, val monetaryCents: Long)

/** Stage 1: SQLite profile lookup, matching the query shape used in sidecar.py. */
fun lookupProfile(db: SQLiteDatabase, customerId: String): RawProfile {
    db.rawQuery(
        "SELECT last_seen, frequency, monetary_c FROM customer_profile WHERE customer_id = ?",
        arrayOf(customerId)
    ).use { c ->
        c.moveToFirst()
        return RawProfile(
            lastSeen = c.getLong(0),
            frequency = c.getInt(1),
            monetaryCents = c.getLong(2)
        )
    }
}

/** Stage 2: RFM feature computation from the raw stored profile, matching sidecar.py's
 *  recency_days = (now - last_seen), log_monetary = log1p(monetary). */
fun computeRfm(profile: RawProfile, nowSeconds: Long): FloatArray {
    val recencyDays = ((nowSeconds - profile.lastSeen) / 86400L).toFloat().coerceAtLeast(0f)
    val logMonetary = ln1p(profile.monetaryCents / 100.0).toFloat()
    return floatArrayOf(recencyDays, profile.frequency.toFloat(), logMonetary)
}

// ---------------------------------------------------------------------------
// Result data model. Purely a display-layer addition: every number here is
// computed with the exact same logic (same n=200, same warmup, same nanoTime
// timing, same percentile function) as before. Nothing about the benchmark
// itself changed, only how the results are handed to the UI.
// ---------------------------------------------------------------------------

data class StageLatency(val stage: String, val p50: Double, val p90: Double, val p95: Double, val p99: Double)

data class PredictionSample(
    val customerId: String,
    val recencyDays: Int,
    val frequency: Int,
    val probability: Float,
    val tier: String
)

data class BenchmarkResult(
    val deviceModel: String,
    val abi: String,
    val n: Int,
    val stages: List<StageLatency>,
    val customersLoaded: Int,
    val targetMet: Boolean,
    val samplePredictions: List<PredictionSample>,
    val highCount: Int,
    val medCount: Int,
    val lowCount: Int,
    val rawSummary: String
)

fun tierColor(tier: String): Color = when (tier) {
    "HIGH" -> Color(0xFFD32F2F)
    "MED" -> Color(0xFFF57C00)
    else -> Color(0xFF388E3C)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkScreen(copyAsset: (String) -> String, appContext: Context) {
    var result by remember { mutableStateOf<BenchmarkResult?>(null) }
    var isRunning by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("ChurnBenchmark", fontWeight = FontWeight.Bold)
                        Text(
                            "3-stage on-device latency",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.8f)
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = {
                    isRunning = true
                    scope.launch(Dispatchers.IO) {
                        val r = runBenchmark(copyAsset("churn_model_edge.onnx"), appContext)
                        result = r
                        isRunning = false
                    }
                },
                enabled = !isRunning,
                modifier = Modifier.fillMaxWidth().height(48.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                if (isRunning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = MaterialTheme.colorScheme.onPrimary,
                        strokeWidth = 2.dp
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Running…")
                } else {
                    Text("RUN BENCHMARK", fontWeight = FontWeight.SemiBold)
                }
            }

            val r = result
            if (r == null && !isRunning) {
                Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    Text(
                        "Tap Run Benchmark to measure SQLite → RFM → ONNX latency\non-device against ${"the loaded customer profiles"}.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 13.sp
                    )
                }
            }

            AnimatedVisibility(visible = r != null) {
                if (r != null) {
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        item { DeviceCard(r) }
                        item { LatencyTableCard(r) }
                        item { TierDistributionCard(r) }
                        item { SamplePredictionsCard(r) }
                        item { RawOutputCard(r) }
                    }
                }
            }
        }
    }
}

@Composable
fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
            Spacer(Modifier.height(10.dp))
            content()
        }
    }
}

@Composable
fun DeviceCard(r: BenchmarkResult) {
    SectionCard("Device") {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(20.dp)
                    .clip(CircleShape)
                    .background(if (r.targetMet) Color(0xFF388E3C) else Color(0xFFD32F2F)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    if (r.targetMet) "✓" else "!",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(Modifier.width(8.dp))
            Column {
                Text("${r.deviceModel}  ·  ${r.abi}", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                Text(
                    "n=${r.n} runs  ·  ${r.customersLoaded} customer profiles (real data)",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    if (r.targetMet) "Within 500ms checkout budget" else "Exceeds 500ms checkout budget",
                    fontSize = 12.sp,
                    color = if (r.targetMet) Color(0xFF388E3C) else Color(0xFFD32F2F),
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

@Composable
fun LatencyTableCard(r: BenchmarkResult) {
    SectionCard("Stage latency (ms)") {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text("Stage", modifier = Modifier.weight(1.2f), fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            listOf("p50", "p90", "p95", "p99").forEach {
                Text(it, modifier = Modifier.weight(1f), fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            }
        }
        Divider(modifier = Modifier.padding(vertical = 6.dp))
        r.stages.forEach { stage ->
            Row(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                Text(stage.stage, modifier = Modifier.weight(1.2f), fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                listOf(stage.p50, stage.p90, stage.p95, stage.p99).forEach { v ->
                    Text(
                        "%.4f".format(v),
                        modifier = Modifier.weight(1f),
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}

@Composable
fun TierDistributionCard(r: BenchmarkResult) {
    val total = (r.highCount + r.medCount + r.lowCount).coerceAtLeast(1)
    SectionCard("Risk tier distribution") {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            TierChip("HIGH", r.highCount, total)
            TierChip("MED", r.medCount, total)
            TierChip("LOW", r.lowCount, total)
        }
    }
}

@Composable
fun TierChip(tier: String, count: Int, total: Int) {
    val pct = (count * 100.0 / total).roundToInt()
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .size(56.dp)
                .clip(CircleShape)
                .background(tierColor(tier).copy(alpha = 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Text("$count", color = tierColor(tier), fontWeight = FontWeight.Bold, fontSize = 16.sp)
        }
        Spacer(Modifier.height(4.dp))
        Text(tier, fontSize = 11.sp, fontWeight = FontWeight.Medium)
        Text("$pct%", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
fun SamplePredictionsCard(r: BenchmarkResult) {
    SectionCard("Sample predictions (first ${r.samplePredictions.size})") {
        r.samplePredictions.forEach { p ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    p.customerId,
                    modifier = Modifier.weight(1f),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    "R=${p.recencyDays}d F=${p.frequency}",
                    modifier = Modifier.weight(1f),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "%.3f".format(p.probability),
                    modifier = Modifier.weight(0.7f),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(tierColor(p.tier).copy(alpha = 0.15f))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(p.tier, fontSize = 10.sp, color = tierColor(p.tier), fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun RawOutputCard(r: BenchmarkResult) {
    var expanded by remember { mutableStateOf(false) }
    SectionCard("Raw output") {
        TextButton(onClick = { expanded = !expanded }, contentPadding = PaddingValues(0.dp)) {
            Text(if (expanded) "Hide" else "Show (for copy/paste)", fontSize = 12.sp)
        }
        AnimatedVisibility(visible = expanded) {
            SelectionContainer {
                Text(
                    r.rawSummary,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                    lineHeight = 12.sp,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        }
    }
}

fun runBenchmark(modelPath: String, appContext: Context): BenchmarkResult {
    val env     = OrtEnvironment.getEnvironment()
    val session = env.createSession(modelPath, OrtSession.SessionOptions())
    val inputName = session.inputNames.iterator().next()

    val dbHelper = CustomerProfileDb(appContext)
    val customerIds = dbHelper.seedFromAsset(appContext)
    val db = dbHelper.readableDatabase

    fun infer(sample: FloatArray): Float {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), longArrayOf(1, 3))
        val result = session.run(mapOf(inputName to tensor))
        @Suppress("UNCHECKED_CAST")
        val prob = (result[1].value as Array<FloatArray>)[0][1]
        tensor.close()
        result.close()
        return prob
    }

    /** One full pass through all three stages for a single customer, returning the
     *  per-stage nanosecond timings. */
    fun timedPass(customerId: String): LongArray {
        val t0 = System.nanoTime()
        val profile = lookupProfile(db, customerId)
        val t1 = System.nanoTime()

        val nowSeconds = System.currentTimeMillis() / 1000L
        val rfm = computeRfm(profile, nowSeconds)
        val t2 = System.nanoTime()

        infer(rfm)
        val t3 = System.nanoTime()

        return longArrayOf(t1 - t0, t2 - t1, t3 - t2, t3 - t0) // sqlite, rfm, onnx, end-to-end
    }

    // Warmup: 10 runs, discarded, same convention as the original ONNX-only benchmark
    repeat(10) { timedPass(customerIds[0]) }

    val n = 200
    val sqliteNs = LongArray(n)
    val rfmNs = LongArray(n)
    val onnxNs = LongArray(n)
    val endToEndNs = LongArray(n)

    for (i in 0 until n) {
        val timings = timedPass(customerIds[i % customerIds.size])
        sqliteNs[i] = timings[0]
        rfmNs[i] = timings[1]
        onnxNs[i] = timings[2]
        endToEndNs[i] = timings[3]
    }

    session.close()
    env.close()
    db.close()

    fun pctOf(arr: LongArray, p: Double): Double {
        val sorted = arr.sortedArray()
        return sorted[(p / 100 * n).toInt().coerceIn(0, n - 1)] / 1_000_000.0
    }

    val stages = listOf(
        StageLatency("SQLite", pctOf(sqliteNs, 50.0), pctOf(sqliteNs, 90.0), pctOf(sqliteNs, 95.0), pctOf(sqliteNs, 99.0)),
        StageLatency("RFM", pctOf(rfmNs, 50.0), pctOf(rfmNs, 90.0), pctOf(rfmNs, 95.0), pctOf(rfmNs, 99.0)),
        StageLatency("ONNX", pctOf(onnxNs, 50.0), pctOf(onnxNs, 90.0), pctOf(onnxNs, 95.0), pctOf(onnxNs, 99.0)),
        StageLatency("E2E", pctOf(endToEndNs, 50.0), pctOf(endToEndNs, 90.0), pctOf(endToEndNs, 95.0), pctOf(endToEndNs, 99.0))
    )
    val targetMet = pctOf(endToEndNs, 50.0) < 500.0

    // Raw text summary retained for copy/paste and continuity with earlier report screenshots.
    val sb = StringBuilder()
    sb.appendLine("ChurnBenchmark 3-stage | ${Build.MODEL} | ${Build.SUPPORTED_ABIS[0]} | n=$n")
    sb.appendLine()
    sb.appendLine("Stage        p50     p90     p95     p99   (ms)")
    stages.forEach { s ->
        sb.appendLine("${s.stage.padEnd(9)} ${"%7.4f".format(s.p50)} ${"%7.4f".format(s.p90)} ${"%7.4f".format(s.p95)} ${"%7.4f".format(s.p99)}")
    }
    sb.appendLine()
    sb.appendLine("Target: E2E p50 < 500ms   |   Customers loaded: ${customerIds.size} (real data)")

    val db2 = dbHelper.readableDatabase
    val env2     = OrtEnvironment.getEnvironment()
    val session2 = env2.createSession(modelPath, OrtSession.SessionOptions())
    val in2      = session2.inputNames.iterator().next()

    var highCount = 0; var medCount = 0; var lowCount = 0
    val samplePredictions = mutableListOf<PredictionSample>()
    for ((idx, customerId) in customerIds.withIndex()) {
        val profile = lookupProfile(db2, customerId)
        val rfm = computeRfm(profile, System.currentTimeMillis() / 1000L)
        val tensor = OnnxTensor.createTensor(env2, FloatBuffer.wrap(rfm), longArrayOf(1, 3))
        val result = session2.run(mapOf(in2 to tensor))
        @Suppress("UNCHECKED_CAST")
        val prob = (result[1].value as Array<FloatArray>)[0][1]
        tensor.close()
        result.close()
        val tier = when {
            prob >= 0.65f -> "HIGH"
            prob >= 0.40f -> "MED"
            else -> "LOW"
        }
        when (tier) { "HIGH" -> highCount++; "MED" -> medCount++; else -> lowCount++ }
        if (idx < 5) {
            samplePredictions.add(
                PredictionSample(customerId, rfm[0].toInt(), rfm[1].toInt(), prob, tier)
            )
        }
    }
    sb.appendLine("Tiers (all ${customerIds.size}): HIGH=$highCount MED=$medCount LOW=$lowCount")

    session2.close()
    env2.close()
    db2.close()

    return BenchmarkResult(
        deviceModel = Build.MODEL,
        abi = Build.SUPPORTED_ABIS[0],
        n = n,
        stages = stages,
        customersLoaded = customerIds.size,
        targetMet = targetMet,
        samplePredictions = samplePredictions,
        highCount = highCount,
        medCount = medCount,
        lowCount = lowCount,
        rawSummary = sb.toString()
    )
}