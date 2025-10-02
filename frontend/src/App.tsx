import { useState } from 'react';
import { AppShell } from './components/AppShell';
import { Home } from './components/Home';
import { Datasets } from './components/Datasets';
import { LightCurves } from './components/LightCurves';
import { ModelLab } from './components/ModelLab';
import { Predict } from './components/Predict';
import { Reset } from './components/Reset';
import { PlaceholderSection } from './components/PlaceholderSection';
import { BarChart3, FileText, Book, User, Trash2 } from 'lucide-react';

export default function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [isDark, setIsDark] = useState(false);

  const handleThemeToggle = () => {
    setIsDark(!isDark);
  };

  const renderSection = () => {
    switch (activeSection) {
      case 'home':
        return <Home onSectionChange={setActiveSection} />;
      case 'datasets':
        return <Datasets />;
      case 'light-curves':
        return <LightCurves />;
      case 'model-lab':
        return <ModelLab />;
      case 'experiments':
        return (
          <PlaceholderSection
            title="Experiments"
            description="Track and manage your machine learning experiments"
            icon={BarChart3}
          />
        );
      case 'predict':
        return <Predict />;
      case 'reset':
        return <Reset />;
      case 'reports':
        return (
          <PlaceholderSection
            title="Reports"
            description="Generate exportable summaries and documentation"
            icon={FileText}
          />
        );
      case 'docs':
        return (
          <PlaceholderSection
            title="Docs & Learn"
            description="Guides, tutorials, and exoplanet glossary"
            icon={Book}
          />
        );
      case 'account':
        return (
          <PlaceholderSection
            title="Account"
            description="Manage your profile, preferences, and API keys"
            icon={User}
          />
        );
      default:
        return <Home onSectionChange={setActiveSection} />;
    }
  };

  return (
    <AppShell
      activeSection={activeSection}
      onSectionChange={setActiveSection}
      isDark={isDark}
      onThemeToggle={handleThemeToggle}
    >
      {renderSection()}
    </AppShell>
  );
}