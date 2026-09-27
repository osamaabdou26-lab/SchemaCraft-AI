"""Quick-start domain descriptions shown as preset buttons in the UI."""

PRESETS: dict[str, str] = {
    "🛒 E-Commerce Order Pipeline": (
        "An e-commerce order. Each order has a UUID, a customer (id, full name, email, "
        "optional phone), a shipping address (street, city, postal code, ISO 3166-1 "
        "alpha-2 country code), and 1 to 50 line items. Each line item has a SKU "
        "(pattern ABC-12345), product name, quantity (1-100) and unit price in USD "
        "(non-negative, 2 decimals). The order has a status that is one of pending, "
        "paid, shipped, delivered, cancelled or refunded, a payment method (card, "
        "paypal, bank_transfer), a created_at timestamp, and an order total."
    ),
    "🎓 University Student Portal": (
        "A university student portal record. A student has a student ID (pattern "
        "S followed by 7 digits), first and last name, university email, date of birth, "
        "enrollment status (active, on_leave, graduated, withdrawn), a major, and a GPA "
        "between 0.0 and 4.0. The student has enrollments; each enrollment references a "
        "course (course code like CS101, title, credit hours 1-6, instructor name) with "
        "a semester (Fall/Spring/Summer plus year) and a grade that is one of A, B, C, D, "
        "F, or null when in progress."
    ),
    "💳 SaaS Billing & Subscriptions": (
        "A SaaS subscription account. It has an account ID, company name, billing email, "
        "a plan (free, starter, pro, enterprise), billing interval (monthly or yearly), "
        "seat count (1-10000), a subscription status (trialing, active, past_due, "
        "canceled), trial end date (nullable), current period start and end dates, and "
        "a list of invoices. Each invoice has an invoice number, amount due in cents, "
        "currency (ISO 4217 code), status (draft, open, paid, void, uncollectible), "
        "issued_at and optional paid_at timestamps."
    ),
    "📡 IoT Sensor Logs": (
        "An IoT telemetry batch from a single device. It has a device ID, firmware "
        "version (semantic version string), device type (thermostat, air_quality, "
        "motion, smart_meter), GPS location (latitude -90..90, longitude -180..180), "
        "a battery level percentage, and a list of 1-100 readings. Each reading has a "
        "timestamp, metric name (temperature_c, humidity_pct, co2_ppm, power_w), a "
        "numeric value, and a quality flag (good, suspect, bad)."
    ),
}
