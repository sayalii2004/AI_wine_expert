import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: BASE, timeout: 15000 });

export const predict = (features, wine_type) =>
  api.post('/api/predict', { features, wine_type }).then(r => r.data);

export const explain = (features, wine_type) =>
  api.post('/api/explain', { features, wine_type }).then(r => r.data);

export const recommend = (features, wine_type) =>
  api.post('/api/recommend', { features, wine_type }).then(r => r.data);

export const getGlobalImportance = () =>
  api.get('/api/analytics/global-importance').then(r => r.data);

export const getGradeStats = (wine_type, grade) =>
  api.get('/api/analytics/grade-stats', { params: { wine_type, grade } }).then(r => r.data);

export const getModelMetrics = () =>
  api.get('/api/analytics/model-metrics').then(r => r.data);