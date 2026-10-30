"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, CircleMarker, MapContainer, Marker, Polygon, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export type MapRiskPoint = {
  lat: number;
  lon: number;
  riskPercent?: number;
  riskLabel?: "LOW" | "MODERATE" | "HIGH" | string;
  probability?: number;
  aiAnalysis?: string;
  updatedAt?: string;
};

type MapPickerProps = {
  lat: number;
  lon: number;
  onChange: (lat: number, lon: number) => void;
  areaRadiusMeters?: number;
  showRadiusArea?: boolean;
  polygonVertices?: Array<[number, number]>;
  sampledPoints?: Array<[number, number]>;
  riskPoints?: MapRiskPoint[];
  currentPointRisk?: {
    probability: number;
    riskLabel: "LOW" | "MODERATE" | "HIGH" | string;
  } | null;
  mapHeightClass?: string;
};

type SearchResult = {
  display_name: string;
  lat: string;
  lon: string;
};

// Fix default marker icons when bundling with Next.js.
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function RecenterMap({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  const previousRef = useRef<[number, number] | null>(null);

  useEffect(() => {
    const previous = previousRef.current;
    const next: [number, number] = [lat, lon];
    if (!previous) {
      previousRef.current = next;
      map.setView(next, map.getZoom(), { animate: false });
      return;
    }

    const latDiff = Math.abs(previous[0] - lat);
    const lonDiff = Math.abs(previous[1] - lon);
    if (latDiff < 1e-7 && lonDiff < 1e-7) {
      return;
    }

    previousRef.current = next;
    map.flyTo(next, map.getZoom(), { animate: true, duration: 0.45 });
  }, [lat, lon, map]);

  return null;
}

function ClickHandler({ onChange }: { onChange: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(event) {
      onChange(event.latlng.lat, event.latlng.lng);
    },
  });
  return null;
}

export default function MapPicker({
  lat,
  lon,
  onChange,
  areaRadiusMeters = 300,
  showRadiusArea = false,
  polygonVertices = [],
  sampledPoints = [],
  riskPoints = [],
  currentPointRisk = null,
  mapHeightClass = "h-[240px]",
}: MapPickerProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [showHeatmapLayer, setShowHeatmapLayer] = useState(true);
  const [showMarkersLayer, setShowMarkersLayer] = useState(true);

  const markerPosition = useMemo(() => [lat, lon] as [number, number], [lat, lon]);
  const activeMarkerIcon = useMemo(
    () =>
      L.divIcon({
        className: "geo-pulse-marker",
        html: '<span class="geo-pulse-dot"></span>',
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      }),
    []
  );

  async function handleSearch() {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setSearching(true);
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(query.trim())}`;
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });
      const data = (await response.json()) as SearchResult[];
      setResults(Array.isArray(data) ? data : []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  // Has genuine spatial risk data
  const hasSpatialRiskData = riskPoints.length > 0 || currentPointRisk !== null;

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-wide text-dim">Geospatial Explorer & Risk Map</p>
        {/* Layer Controls */}
        <div className="flex items-center gap-1.5 text-[10px]">
          <button
            type="button"
            onClick={() => setShowHeatmapLayer((prev) => !prev)}
            className={`rounded-md border px-2 py-0.5 transition ${
              showHeatmapLayer
                ? "border-red-400/40 bg-red-500/15 font-semibold text-red-200"
                : "border-white/10 bg-white/5 text-zinc-400 hover:text-zinc-200"
            }`}
            title="Toggle spatial erosion risk heatmap layer"
          >
            🔥 Heatmap: {showHeatmapLayer ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            onClick={() => setShowMarkersLayer((prev) => !prev)}
            className={`rounded-md border px-2 py-0.5 transition ${
              showMarkersLayer
                ? "border-cyan-400/40 bg-cyan-500/15 font-semibold text-cyan-200"
                : "border-white/10 bg-white/5 text-zinc-400 hover:text-zinc-200"
            }`}
            title="Toggle risk point markers"
          >
            📍 Markers: {showMarkersLayer ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      <div className="flex gap-2">
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search location (e.g. Hampi, Pattadakal)"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleSearch();
            }
          }}
        />
        <button type="button" className="button-primary px-3 py-2" onClick={handleSearch}>
          {searching ? "..." : "Go"}
        </button>
      </div>

      {results.length > 0 ? (
        <div className="max-h-32 overflow-auto rounded-xl border border-borderSoft bg-slate-900/80">
          {results.map((item) => (
            <button
              key={`${item.lat}-${item.lon}-${item.display_name}`}
              type="button"
              className="block w-full border-b border-borderSoft px-3 py-2 text-left text-xs text-dim hover:bg-white/5"
              onClick={() => {
                onChange(Number(item.lat), Number(item.lon));
                setResults([]);
                setQuery(item.display_name);
              }}
            >
              {item.display_name}
            </button>
          ))}
        </div>
      ) : null}

      <div className="relative overflow-hidden rounded-xl border border-borderSoft transition hover:border-cyan-300/35">
        <MapContainer center={markerPosition} zoom={10} scrollWheelZoom preferCanvas className={`${mapHeightClass} w-full`}>
          <TileLayer
            attribution="Tiles &copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />

          {showRadiusArea ? (
            <Circle
              center={markerPosition}
              radius={areaRadiusMeters}
              pathOptions={{ color: "#22d3ee", weight: 2, fillOpacity: 0.12 }}
            />
          ) : null}

          {polygonVertices.length >= 3 ? (
            <Polygon
              positions={polygonVertices}
              pathOptions={{ color: "#f59e0b", weight: 2, fillOpacity: 0.12 }}
            />
          ) : null}

          {polygonVertices.map((vertex, index) => (
            <CircleMarker
              key={`vertex-${index}-${vertex[0]}-${vertex[1]}`}
              center={vertex}
              radius={5}
              pathOptions={{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 0.9, weight: 1 }}
            />
          ))}

          {/* Genuine Single-Point Erosion Risk Heatmap (when analyzed) */}
          {showHeatmapLayer && currentPointRisk ? (() => {
            const prob = currentPointRisk.probability;
            const color = prob >= 0.7 ? "#ef4444" : prob >= 0.3 ? "#f59e0b" : "#10b981";
            return (
              <>
                <Circle
                  center={markerPosition}
                  radius={Math.max(160, areaRadiusMeters * 0.6)}
                  pathOptions={{ color: "transparent", fillColor: color, fillOpacity: 0.18, weight: 0 }}
                />
                <Circle
                  center={markerPosition}
                  radius={Math.max(75, areaRadiusMeters * 0.28)}
                  pathOptions={{ color: "transparent", fillColor: color, fillOpacity: 0.32, weight: 0 }}
                />
              </>
            );
          })() : null}

          {/* Multi-Point Spatial Erosion Risk Heatmap */}
          {showHeatmapLayer && riskPoints.map((point, index) => {
            const prob = point.probability ?? ((point.riskPercent ?? 50) / 100);
            const color = prob >= 0.7 ? "#ef4444" : prob >= 0.3 ? "#f59e0b" : "#10b981";
            const radius = Math.max(30, areaRadiusMeters * 0.3);
            return (
              <span key={`heat-zone-${index}-${point.lat}-${point.lon}`}>
                <Circle
                  center={[point.lat, point.lon]}
                  radius={radius}
                  pathOptions={{ color: "transparent", fillColor: color, fillOpacity: 0.20, weight: 0 }}
                />
                <Circle
                  center={[point.lat, point.lon]}
                  radius={Math.max(15, radius * 0.5)}
                  pathOptions={{ color: "transparent", fillColor: color, fillOpacity: 0.36, weight: 0 }}
                />
              </span>
            );
          })}

          {/* Markers / Tooltips */}
          {showMarkersLayer && (riskPoints.length > 0
            ? riskPoints.map((point, index) => {
                const prob = point.probability ?? ((point.riskPercent ?? 50) / 100);
                const color = prob >= 0.7 ? "#ef4444" : prob >= 0.3 ? "#f59e0b" : "#10b981";
                const strokeColor = prob >= 0.7 ? "#b91c1c" : prob >= 0.3 ? "#d97706" : "#047857";
                return (
                  <CircleMarker
                    key={`risk-pt-${index}-${point.lat}-${point.lon}`}
                    center={[point.lat, point.lon]}
                    radius={5}
                    pathOptions={{ color: strokeColor, fillColor: color, fillOpacity: 0.95, weight: 1.5 }}
                  >
                    <Tooltip direction="top" offset={[0, -8]} opacity={1} className="leaf-point-label">
                      <div className="text-center">
                        <span className="font-bold text-white">{point.riskPercent ?? Math.round(prob * 100)}% Risk</span>
                        <span className="block text-[10px] text-zinc-300">({point.riskLabel || "MODERATE"})</span>
                      </div>
                    </Tooltip>
                  </CircleMarker>
                );
              })
            : sampledPoints.map((point, index) => (
                <CircleMarker
                  key={`sample-${index}-${point[0]}-${point[1]}`}
                  center={point}
                  radius={4}
                  pathOptions={{ color: "#10b981", fillColor: "#10b981", fillOpacity: 0.85, weight: 1 }}
                >
                  <Tooltip direction="top" offset={[0, -8]} opacity={1} className="leaf-point-label">
                    Point {index + 1}
                  </Tooltip>
                </CircleMarker>
              ))
          )}

          <Marker position={markerPosition} icon={activeMarkerIcon} />
          <ClickHandler onChange={onChange} />
          <RecenterMap lat={lat} lon={lon} />
        </MapContainer>
      </div>

      {/* Heatmap Status & Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-0.5 text-[11px] text-dim">
        <p>
          Coordinates: <span className="font-mono text-zinc-200">{lat.toFixed(6)}, {lon.toFixed(6)}</span>
        </p>

        {showHeatmapLayer && (
          <div className="flex items-center gap-3 text-[10px]">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
              Low (&lt;30%)
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50" />
              Moderate (30-69%)
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
              High (&ge;70%)
            </span>
          </div>
        )}
      </div>

      {showHeatmapLayer && !hasSpatialRiskData && (
        <p className="rounded-lg border border-amber-400/20 bg-amber-500/10 px-2.5 py-1.5 text-[11px] text-amber-200/90">
          ℹ️ <strong>Erosion Risk Heatmap:</strong> No spatial risk data available yet. Click <em>Analyze Terrain</em> or sample map points to render the live risk intensity overlay.
        </p>
      )}
    </div>
  );
}
