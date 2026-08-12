const express = require('express');
const multer  = require('multer');
const { Kafka } = require('kafkajs');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.json());

// Local uploads directory
const uploadDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
const upload = multer({ storage });

// Connect to local Kafka
const kafka = new Kafka({
  clientId: 'ingest-service',
  brokers: [process.env.KAFKA_BROKER || 'localhost:9092']
});
const producer = kafka.producer();

app.post('/ingest', upload.single('image'), async (req, res) => {
  try {
    await producer.connect();

    const payload = {
      eventId: `evt_${Date.now()}`,
      imagePath: req.file ? req.file.path : null,
      numericData: {
        temperature: parseFloat(req.body.temperature || 0),
        pressure: parseFloat(req.body.pressure || 0),
        sensorValue: parseFloat(req.body.sensorValue || 0)
      },
      timestamp: new Date().toISOString()
    };

    await producer.send({
      topic: 'raw-dataset-events',
      messages: [{ value: JSON.stringify(payload) }]
    });

    console.log('Ingested Image + Numerical Event:', payload);
    res.status(202).json({ status: 'Accepted', payload });
  } catch (err) {
    console.error('Ingestion error:', err);
    res.status(500).json({ error: 'Ingestion failed' });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Ingest service listening on port ${PORT}`));
