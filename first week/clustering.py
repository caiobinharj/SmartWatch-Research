from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def read_arff_30_features(file_path: str) -> tuple[list[str], np.ndarray]:
	data_started = False
	activities: list[str] = []
	features: list[list[float]] = []
	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("%"):
				continue
			if not data_started:
				if line.lower() == "@data":
					data_started = True
				continue
			# After @data: CSV rows
			parts = [p.strip() for p in line.split(",")]
			if not parts:
				continue
			act = parts[0]
			vals = parts[1:31]
			if len(vals) < 30:
				continue
			try:
				vec = [float(v) for v in vals]
			except ValueError:
				continue
			activities.append(act)
			features.append(vec)
	return activities, np.asarray(features, dtype=float)


def find_id_pairs(accel_dir: str, gyro_dir: str) -> list[tuple[str, str, str]]:
	pairs: list[tuple[str, str, str]] = []
	acc_dir = Path(accel_dir)
	gyr_dir = Path(gyro_dir)
	for acc_path in acc_dir.glob("data_*_accel_watch.arff"):
		name = acc_path.stem  # e.g., data_1600_accel_watch
		parts = name.split("_")
		if len(parts) < 4 or parts[0] != "data" or parts[2] != "accel" or parts[3] != "watch":
			continue
		person_id = parts[1]
		gyro_path = gyr_dir / f"data_{person_id}_gyro_watch.arff"
		if gyro_path.exists():
			pairs.append((person_id, str(acc_path), str(gyro_path)))
	return sorted(pairs, key=lambda x: int(x[0]))


def build_dataset(accel_dir: str, gyro_dir: str) -> tuple[np.ndarray, list[str], list[tuple[str, int]]]:
	X_all: list[np.ndarray] = []
	acts_all: list[str] = []
	index_meta: list[tuple[str, int]] = []  # (person_id, window_index)
	for person_id, acc_path, gyr_path in find_id_pairs(accel_dir, gyro_dir):
		acts_a, fa = read_arff_30_features(acc_path)
		acts_g, fg = read_arff_30_features(gyr_path)
		if fa.size == 0 or fg.size == 0:
			continue
		n = min(fa.shape[0], fg.shape[0])
		fa = fa[:n]
		fg = fg[:n]
		# Prefer activity from accel (should match gyro)
		acts = acts_a[:n]
		X = np.concatenate([fa, fg], axis=1)
		X_all.append(X)
		acts_all.extend(acts)
		index_meta.extend([(person_id, i) for i in range(n)])
	if not X_all:
		return np.empty((0, 60)), [], []
	return np.vstack(X_all), acts_all, index_meta


def kmeans_cluster(X: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> tuple[np.ndarray, KMeans, StandardScaler]:
	scaler = StandardScaler()
	Xn = scaler.fit_transform(X)
	model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
	labels = model.fit_predict(Xn)
	return labels, model, scaler


def activity_colors(acts: list[str]) -> list[str]:
	groups = {
		'red': set('ABC'),
		'blue': set('DE'),
		'#90EE90': set('FGPQR'),
		'yellow': set('HIJKL'),
		'#FF1493': set('MNO'),
	}
	def pick(a: str) -> str:
		for c, s in groups.items():
			if a in s:
				return c
		return 'gray'
	return [pick(a) for a in acts]


def plot_time_clusters_global(output_dir: str, labels: np.ndarray, acts: list[str], index_meta: list[tuple[str, int]]):
	# Concatena as séries de cada ID no tempo (10s por janela)
	by_id: dict[str, list[tuple[int, int, str]]] = {}
	for (pid, widx), lab, act in zip(index_meta, labels, acts):
		by_id.setdefault(pid, []).append((widx, lab, act))
	t_global: list[float] = []
	y_global: list[int] = []
	colors: list[str] = []
	offset = 0.0
	for pid in sorted(by_id.keys(), key=lambda k: int(k)):
		seq = sorted(by_id[pid], key=lambda t: t[0])
		times = [w * 10.0 + offset for w, _, _ in seq]
		labs = [lab for _, lab, _ in seq]
		acts_seq = [act for _, _, act in seq]
		cols = activity_colors(acts_seq)
		t_global.extend(times)
		y_global.extend(labs)
		colors.extend(cols)
		if times:
			offset = times[-1] + 10.0
	plt.figure(figsize=(14, 4))
	plt.scatter(t_global, y_global, c=colors, s=8)
	plt.yticks([0,1,2,3,4], ["C0","C1","C2","C3","C4"])  # 5 clusters
	plt.ylim(-0.5, 4.5)
	plt.xlabel("Tempo (s)")
	plt.ylabel("Cluster")
	plt.title("Tempo x Cluster (k=5)")
	plt.grid(axis='y', linestyle='--', alpha=0.4)
	plt.tight_layout()
	out_path = Path(output_dir) / "clusters_time_global.png"
	plt.savefig(str(out_path), dpi=150)
	plt.close()


def main():
	base_dir = Path(__file__).resolve().parent
	accel_dir = str(base_dir / "watch" / "accel")
	gyro_dir = str(base_dir / "watch" / "gyro")
	output_dir = str(base_dir / "watch")
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	X, acts, index_meta = build_dataset(accel_dir, gyro_dir)
	if X.shape[0] == 0:
		print("Nenhum dado encontrado para clusterização.")
		return
	labels, model, scaler = kmeans_cluster(X, n_clusters=5, random_state=42)
	plot_time_clusters_global(output_dir, labels, acts, index_meta)
	print(f"Total de pontos: {X.shape[0]} | Dimensões: {X.shape[1]} | Figura: watch/clusters_time_global.png")


if __name__ == "__main__":
	main()


