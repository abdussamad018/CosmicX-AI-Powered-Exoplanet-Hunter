import { useState } from 'react';
import { Button } from './ui/button';
import { Sheet, SheetContent, SheetTrigger } from './ui/sheet';
import { 
  Home, 
  Database, 
  Activity, 
  FlaskConical, 
  BarChart3, 
  Zap, 
  FileText, 
  Book, 
  User, 
  Search, 
  Bell, 
  HelpCircle, 
  Sun, 
  Moon, 
  Menu,
  ChevronLeft,
  ChevronRight,
  Trash2
} from 'lucide-react';
import { cn } from './ui/utils';

interface AppShellProps {
  children: React.ReactNode;
  activeSection: string;
  onSectionChange: (section: string) => void;
  isDark: boolean;
  onThemeToggle: () => void;
}

const navigationItems = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'light-curves', label: 'Light Curves', icon: Activity },
  { id: 'model-lab', label: 'Model Lab', icon: FlaskConical },
  { id: 'experiments', label: 'Experiments', icon: BarChart3 },
  { id: 'predict', label: 'Predict', icon: Zap },
  { id: 'reset', label: 'Reset Data', icon: Trash2 },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'docs', label: 'Docs & Learn', icon: Book },
  { id: 'account', label: 'Account', icon: User },
];

export function AppShell({ children, activeSection, onSectionChange, isDark, onThemeToggle }: AppShellProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  return (
    <div className={cn("h-screen flex", isDark && "dark")}>
      {/* Desktop Sidebar */}
      <aside className={cn(
        "hidden lg:flex flex-col bg-sidebar border-r border-sidebar-border transition-all duration-300",
        isCollapsed ? "w-16" : "w-72"
      )}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
          {!isCollapsed && (
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold text-sidebar-foreground">ExoScope</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-sidebar-foreground hover:bg-sidebar-accent"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2">
          <ul className="space-y-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              
              return (
                <li key={item.id}>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                      isActive && "bg-sidebar-accent text-sidebar-accent-foreground",
                      isCollapsed && "px-2"
                    )}
                    onClick={() => onSectionChange(item.id)}
                  >
                    <Icon className={cn("w-5 h-5", !isCollapsed && "mr-3")} />
                    {!isCollapsed && <span>{item.label}</span>}
                  </Button>
                </li>
              );
            })}
          </ul>
        </nav>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <header className="bg-background border-b border-border p-4 flex items-center justify-between">
          {/* Mobile Menu */}
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="sm" className="lg:hidden">
                <Menu className="w-5 h-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72 p-0">
              <div className="flex items-center space-x-2 p-4 border-b">
                <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                  <Activity className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="text-lg font-semibold">ExoScope</span>
              </div>
              <nav className="p-2">
                <ul className="space-y-1">
                  {navigationItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeSection === item.id;
                    
                    return (
                      <li key={item.id}>
                        <Button
                          variant={isActive ? "secondary" : "ghost"}
                          className="w-full justify-start"
                          onClick={() => onSectionChange(item.id)}
                        >
                          <Icon className="w-5 h-5 mr-3" />
                          <span>{item.label}</span>
                        </Button>
                      </li>
                    );
                  })}
                </ul>
              </nav>
            </SheetContent>
          </Sheet>

          {/* Search and Utilities */}
          <div className="flex-1 max-w-md mx-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <input 
                type="text" 
                placeholder="Search datasets, experiments..."
                className="w-full pl-10 pr-4 py-2 bg-input-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>

          {/* Right Actions */}
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm">
              <HelpCircle className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="sm">
              <Bell className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="sm" onClick={onThemeToggle}>
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </Button>
            <Button variant="ghost" size="sm">
              <User className="w-5 h-5" />
            </Button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}