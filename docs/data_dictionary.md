# Data Dictionary — Olist Brazilian E-Commerce Dataset

> Generated: 2026-09-19T15:08:51.954128
> Time Range: 2016-09-04 21:15:19 → 2018-10-17 17:30:18
> Total Orders: 99,441
> Unique Customers: 96,096
> Total GMV (BRL): R$ 16,008,872.12

---

## Feature Availability Matrix

| Feature | Available | Source | Notes |
|---------|-----------|--------|-------|
| Revenue (GMV) | ✅ | `payments.payment_value` |  |
| Order Volume | ✅ | `orders.order_id` |  |
| Average Order Value (AOV) | ✅ | `payments.payment_value / count(orders)` |  |
| Freight Value | ✅ | `order_items.freight_value` |  |
| Cost of Goods Sold (COGS) | ❌ | `N/A` | LIMITATION: No cost data in dataset. Gross margin cannot be calculated. |
| Gross Profit / Margin | ❌ | `N/A` | LIMITATION: Not in dataset. |
| Customer Identity (unique) | ✅ | `customers.customer_unique_id` | customer_unique_id enables cross-order tracking |
| Customer Location | ✅ | `customers.city, state, zip_code_prefix` |  |
| Customer Acquisition Date | ✅ | `first order_purchase_timestamp per customer` |  |
| Product Category | ✅ | `products.product_category_name + translation` |  |
| Product Dimensions/Weight | ✅ | `products.*_cm, products.product_weight_g` |  |
| Product Price | ✅ | `order_items.price` | Selling price per item |
| Order Status | ✅ | `orders.order_status` | Values: delivered, shipped, canceled, etc. |
| Delivery Timestamps | ✅ | `orders.*_date` | Purchase, approval, carrier, delivery, estimated |
| Cancellations (proxy for returns) | ✅ | `orders.order_status = 'canceled'` | No explicit return data; canceled orders used as proxy |
| Payment Type | ✅ | `payments.payment_type` | credit_card, boleto, voucher, debit_card |
| Installments | ✅ | `payments.payment_installments` |  |
| Review Score | ✅ | `reviews.review_score` | 1-5 stars |
| Marketing Spend | ⚠️ SYNTHETIC | `N/A` | SYNTHETIC: Will generate realistic synthetic data clearly labeled |
| Impressions / Clicks / CTR | ⚠️ SYNTHETIC | `N/A` | SYNTHETIC: Clearly labeled throughout |
| CAC / ROAS / CPC | ⚠️ SYNTHETIC | `N/A` | SYNTHETIC: Clearly labeled throughout |
| Attribution | ⚠️ SYNTHETIC | `N/A` | SYNTHETIC: Rule-based attribution on synthetic data. NOT causal inference. |
| Web Sessions / Traffic | ❌ | `N/A` | LIMITATION: No session data in Olist |
| Cart Abandonment (raw) | ❌ | `N/A` | LIMITATION: No session/cart data. Proxy: orders with status='canceled' after approval. |
| Seller Data | ✅ | `sellers.*` | Seller location only |
| Geolocation | ✅ | `geolocation.lat, lng` | By zip code prefix |

---

## Tables

### `orders` — Master order table — one row per order

- **File**: `olist_orders_dataset.csv`
- **Rows**: 99,441
- **Columns**: 8
- **Duplicate rows**: 0
- **Memory**: 25.85 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `order_id` | object | 0 | 0.0% | 99,441 | e481f51cbdc54678b7cc49136f2d6af7: 1, f01059d0d674e1282df4e8fbbe015aa2: 1, fbc17f0f2a2125054d5ac5c22d2d5120: 1 |
| `customer_id` | object | 0 | 0.0% | 99,441 | 9ef432eb6251297304e76186b10a928d: 1, 413f7e58270a32396af030a075b924be: 1, eb4350b67a0264c67e5e06a038e4afbb: 1 |
| `order_status` | object | 0 | 0.0% | 8 | delivered: 96,478, shipped: 1,107, canceled: 625 |
| `order_purchase_timestamp` | datetime64[ns] | 0 | 0.0% | 98,875 | Range: 2016-09-04 21:15:19 → 2018-10-17 17:30:18 |
| `order_approved_at` | datetime64[ns] | 160 | 0.16% | 90,733 | Range: 2016-09-15 12:16:38 → 2018-09-03 17:40:06 |
| `order_delivered_carrier_date` | datetime64[ns] | 1,783 | 1.79% | 81,018 | Range: 2016-10-08 10:34:01 → 2018-09-11 19:48:28 |
| `order_delivered_customer_date` | datetime64[ns] | 2,965 | 2.98% | 95,664 | Range: 2016-10-11 13:46:32 → 2018-10-17 13:22:46 |
| `order_estimated_delivery_date` | datetime64[ns] | 0 | 0.0% | 459 | Range: 2016-09-30 00:00:00 → 2018-11-12 00:00:00 |

### `order_items` — Line items within each order — one row per item

- **File**: `olist_order_items_dataset.csv`
- **Rows**: 112,650
- **Columns**: 7
- **Duplicate rows**: 0
- **Memory**: 30.98 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `order_id` | object | 0 | 0.0% | 98,666 | 8272b63d03f5f79c56e9e4120aec44ef: 21, 1b15974a0141d54e36626dca3fdc731a: 20, ab14fdcfbe524636d65ee38360e22ce8: 20 |
| `order_item_id` | int64 | 0 | 0.0% | 21 | min=1.0, max=21.0, mean=1.1978 |
| `product_id` | object | 0 | 0.0% | 32,951 | aca2eb7d00ea1a7b8ebd4e68314663af: 527, 99a4788cb24856965c36a24e339b6058: 488, 422879e10f46682990de24d770e7f83d: 484 |
| `seller_id` | object | 0 | 0.0% | 3,095 | 6560211a19b47992c3666cc44a7e94c0: 2,033, 4a3ca9315b744ce9f8e9374361493884: 1,987, 1f50f920176fa81dab994f9023523100: 1,931 |
| `shipping_limit_date` | datetime64[ns] | 0 | 0.0% | 93,318 | Range: 2016-09-19 00:15:34 → 2020-04-09 22:35:08 |
| `price` | float64 | 0 | 0.0% | 5,968 | min=0.85, max=6735.0, mean=120.6537 |
| `freight_value` | float64 | 0 | 0.0% | 6,999 | min=0.0, max=409.68, mean=19.9903 |

### `payments` — Payment details per order — one row per payment installment

- **File**: `olist_order_payments_dataset.csv`
- **Rows**: 103,886
- **Columns**: 5
- **Duplicate rows**: 0
- **Memory**: 17.02 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `order_id` | object | 0 | 0.0% | 99,440 | fa65dad1b0e818e3ccc5cb0e39231352: 29, ccf804e764ed5650cd8759557269dc13: 26, 285c2e15bebd4ac83635ccc563dc71f4: 22 |
| `payment_sequential` | int64 | 0 | 0.0% | 29 | min=1.0, max=29.0, mean=1.0927 |
| `payment_type` | object | 0 | 0.0% | 5 | credit_card: 76,795, boleto: 19,784, voucher: 5,775 |
| `payment_installments` | int64 | 0 | 0.0% | 24 | min=0.0, max=24.0, mean=2.8533 |
| `payment_value` | float64 | 0 | 0.0% | 29,077 | min=0.0, max=13664.08, mean=154.1004 |

### `reviews` — Customer reviews — one row per review

- **File**: `olist_order_reviews_dataset.csv`
- **Rows**: 99,224
- **Columns**: 7
- **Duplicate rows**: 0
- **Memory**: 29.12 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `review_id` | object | 0 | 0.0% | 98,410 | 7b606b0d57b078384f0b58eac1d41d78: 3, dbdf1ea31790c8ecfcc6750525661a9b: 3, 32415bbf6e341d5d517080a796f79b5c: 3 |
| `order_id` | object | 0 | 0.0% | 98,673 | c88b1d1b157a9999ce368f218a407141: 3, 8e17072ec97ce29f0e1f111e598b0c85: 3, df56136b8031ecd28e200bb18e6ddb2e: 3 |
| `review_score` | int64 | 0 | 0.0% | 5 | min=1.0, max=5.0, mean=4.0864 |
| `review_comment_title` | object | 87,656 | 88.34% | 4,527 | Recomendo: 423, recomendo: 345, Bom: 293 |
| `review_comment_message` | object | 58,247 | 58.7% | 36,159 | Muito bom: 230, Bom: 189, muito bom: 122 |
| `review_creation_date` | datetime64[ns] | 0 | 0.0% | 636 | Range: 2016-10-02 00:00:00 → 2018-08-31 00:00:00 |
| `review_answer_timestamp` | datetime64[ns] | 0 | 0.0% | 98,248 | Range: 2016-10-07 18:32:28 → 2018-10-29 12:27:35 |

### `customers` — Customer dimension — one row per customer_id (not unique customers)

- **File**: `olist_customers_dataset.csv`
- **Rows**: 99,441
- **Columns**: 5
- **Duplicate rows**: 0
- **Memory**: 27.88 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `customer_id` | object | 0 | 0.0% | 99,441 | 06b8999e2fba1a1fbc88172c00ba8bc7: 1, c023f30c1147aeb0358474f3b1dbc707: 1, b5cbf43f42281920a175fc99650c91d6: 1 |
| `customer_unique_id` | object | 0 | 0.0% | 96,096 | 8d50f5eadf50201ccdcedfb9e2ac8455: 17, 3e43e6105506432c953e165fb2acf44c: 9, 1b6c7548a2a1f9037c1fd3ddfed95f33: 7 |
| `customer_zip_code_prefix` | int64 | 0 | 0.0% | 14,994 | min=1003.0, max=99990.0, mean=35137.4746 |
| `customer_city` | object | 0 | 0.0% | 4,119 | sao paulo: 15,540, rio de janeiro: 6,882, belo horizonte: 2,773 |
| `customer_state` | object | 0 | 0.0% | 27 | SP: 41,746, RJ: 12,852, MG: 11,635 |

### `products` — Product catalogue — one row per product

- **File**: `olist_products_dataset.csv`
- **Rows**: 32,951
- **Columns**: 9
- **Duplicate rows**: 0
- **Memory**: 6.6 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `product_id` | object | 0 | 0.0% | 32,951 | 1e9e8ef04dbcff4541ed26657ea517e5: 1, d05cc9afc85771f597cf4bc9d8f12546: 1, 71b7afd92c42feab780d5ea512fc7348: 1 |
| `product_category_name` | object | 610 | 1.85% | 73 | cama_mesa_banho: 3,029, esporte_lazer: 2,867, moveis_decoracao: 2,657 |
| `product_name_lenght` | float64 | 610 | 1.85% | 66 | min=5.0, max=76.0, mean=48.4769 |
| `product_description_lenght` | float64 | 610 | 1.85% | 2,960 | min=4.0, max=3992.0, mean=771.4953 |
| `product_photos_qty` | float64 | 610 | 1.85% | 19 | min=1.0, max=20.0, mean=2.189 |
| `product_weight_g` | float64 | 2 | 0.01% | 2,204 | min=0.0, max=40425.0, mean=2276.4725 |
| `product_length_cm` | float64 | 2 | 0.01% | 99 | min=7.0, max=105.0, mean=30.8151 |
| `product_height_cm` | float64 | 2 | 0.01% | 102 | min=2.0, max=105.0, mean=16.9377 |
| `product_width_cm` | float64 | 2 | 0.01% | 95 | min=6.0, max=118.0, mean=23.1967 |

### `sellers` — Seller dimension — one row per seller

- **File**: `olist_sellers_dataset.csv`
- **Rows**: 3,095
- **Columns**: 4
- **Duplicate rows**: 0
- **Memory**: 0.62 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `seller_id` | object | 0 | 0.0% | 3,095 | 3442f8959a84dea7ee197c632cb2df15: 1, e26901d5ab434ce92fd9b5c256820a4e: 1, 7e3f87d16fb353f408d467e74fbd8014: 1 |
| `seller_zip_code_prefix` | int64 | 0 | 0.0% | 2,246 | min=1001.0, max=99730.0, mean=32291.0595 |
| `seller_city` | object | 0 | 0.0% | 611 | sao paulo: 694, curitiba: 127, rio de janeiro: 96 |
| `seller_state` | object | 0 | 0.0% | 23 | SP: 1,849, PR: 349, MG: 244 |

### `geolocation` — Geographic coordinates by zip code prefix

- **File**: `olist_geolocation_dataset.csv`
- **Rows**: 1,000,163
- **Columns**: 5
- **Duplicate rows**: 261831
- **Memory**: 136.59 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `geolocation_zip_code_prefix` | int64 | 0 | 0.0% | 19,015 | min=1001.0, max=99990.0, mean=36574.1665 |
| `geolocation_lat` | float64 | 0 | 0.0% | 717,360 | min=-36.6054, max=45.0659, mean=-21.1762 |
| `geolocation_lng` | float64 | 0 | 0.0% | 717,613 | min=-101.4668, max=121.1054, mean=-46.3905 |
| `geolocation_city` | object | 0 | 0.0% | 8,011 | sao paulo: 135,800, rio de janeiro: 62,151, belo horizonte: 27,805 |
| `geolocation_state` | object | 0 | 0.0% | 27 | SP: 404,268, MG: 126,336, RJ: 121,169 |

### `category_translation` — Portuguese → English category name mapping

- **File**: `product_category_name_translation.csv`
- **Rows**: 71
- **Columns**: 2
- **Duplicate rows**: 0
- **Memory**: 0.01 MB

| Column | Type | Nulls | Null% | Unique | Notes |
|--------|------|-------|-------|--------|-------|
| `product_category_name` | object | 0 | 0.0% | 71 | beleza_saude: 1, alimentos: 1, fashion_esporte: 1 |
| `product_category_name_english` | object | 0 | 0.0% | 71 | health_beauty: 1, food: 1, fashion_sport: 1 |

---

## ⚠️ Explicit Limitations

The following features were requested but are **NOT available** in the Olist dataset:

1. **COGS / Profit Margins**: No cost data exists. Gross margin metrics cannot be calculated from real data.
2. **Marketing Spend / CAC / ROAS / CTR**: No marketing data in Olist. These metrics use **[SYNTHETIC]** data generated for demonstration purposes only.
3. **Web Sessions / Traffic**: No session-level data. Funnel analytics and true cart abandonment cannot be measured.
4. **Cart Abandonment**: No cart or session data. Proxy: orders with `status='canceled'` post-approval.
5. **Low Repeat Purchase Rate**: Olist has ~97% single-purchase customers. This is a real characteristic of the dataset that impacts CLV and churn model design.
6. **Attribution ≠ Causation**: All marketing attribution uses rule-based methods (last-click, linear, time-decay). These represent *attributed* revenue, NOT causal impact.
